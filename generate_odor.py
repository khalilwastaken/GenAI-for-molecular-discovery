import sys
import os
import json
import numpy as np
import torch
import typing
import collections
import functools
from omegaconf import OmegaConf
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, QED
from rdkit import DataStructs

from src.analysis.rdkit_functions import compute_molecular_metrics

# ==============================================================================
# 1. BYPASS SÉCURITÉ & PATHS
# ==============================================================================
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf.base import ContainerMetadata, Node

old_load = torch.load
torch.load = functools.partial(old_load, weights_only=False)

torch.serialization.add_safe_globals([
    DictConfig, ListConfig, ContainerMetadata, Node,
    dict, list, collections.defaultdict,
    typing.Any, typing.Union, typing.List, typing.Dict, typing.Optional
])

project_root = os.getcwd()
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import src.datasets
sys.modules['datasets'] = src.datasets
import src.datasets.odor_dataset
sys.modules['datasets.odor_dataset'] = src.datasets.odor_dataset

from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.datasets.odor_dataset import OdorDataModule, OdorInfos, META_CATEGORY_NAMES, aggregate_to_meta
from src.metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
from src.diffusion.extra_features import ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.analysis.visualization import MolecularVisualization

# ==============================================================================
# 2. PARAMÈTRES DE GÉNÉRATION DE MASSE
# ==============================================================================
CHECKPOINT_PATH = "/home/khalil/outputs/2026-03-16/22-59-24-odor/final_model_manual.ckpt"
RESULTS_DIR = "results"  # Dossier pour sauvegarder les résultats

# Méta-catégorie ciblée (nom ou index)
# Options: floral, fruity, sweet, woody, green, spicy, animal_musk, earthy, citrus, chemical, gourmand, powdery_amber
TARGET_META_CATEGORY = "woody"

BATCH_SIZE = 32  # Augmenté de 16 → 32 pour plus de molécules
NUM_BATCHES = 40  # Augmenté à 40 pour plus de tentatives = 1280 tentatives
FILTER_FRAGMENTS = True
SIMILARITY_THRESHOLD = 0.3  # Abaissé de 0.35 → 0.3 (OPTION 2 - moins strict)

# For novelty: keep molecules whose similarity to nearest train example is below this threshold.
# Lower => more novel (farther from train set). Set to None to disable novelty filtering.
NOVELTY_TANIMOTO = 0.5  # Augmenté de 0.25 → 0.5 (OPTION 2 - plus permissif)
NOVELTY_ONLY = False  # If True, save only molecules with tanimoto < NOVELTY_TANIMOTO (or with no match)

# OPTION 3: Filtrage chimique post-génération
MIN_LOGP = -2
MAX_LOGP = 6  # écarter les molécules ultra-lipophiles
MIN_QED = 0.3  # assoupli de 0.4 → 0.3 (accepter plus de molécules)
MIN_MOLWT = 50  # abaissé de 150 → 50 (accepter petites molécules valides)
MAX_MOLWT = 600  # molécules de taille raisonnable

def filter_by_properties(mol):
    """Filtrer une molécule selon ses propriétés chimiques (OPTION 3).
    
    Retourne (keep, reason) où keep: True si la molécule passe les filtres.
    """
    try:
        if mol is None:
            return False, "no_mol"
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        qed = QED.qed(mol)
        
        # Appliquer les filtres
        if logp < MIN_LOGP or logp > MAX_LOGP:
            return False, f"logp_out_of_range({logp:.2f})"
        if mw < MIN_MOLWT or mw > MAX_MOLWT:
            return False, f"molwt_out_of_range({mw:.1f})"
        if qed < MIN_QED:
            return False, f"qed_too_low({qed:.3f})"
        
        return True, "pass"
    except Exception:
        return False, "filter_error"

def translate_smiles(smiles):
    """Traduire un SMILES en description textuelle de la molécule."""
    if smiles is None:
        return ""
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES"
        
        # Compter les atomes
        atoms = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atoms[symbol] = atoms.get(symbol, 0) + 1
        
        # Compter les cycles aromatiques
        num_rings = Chem.GetSSSR(mol)
        num_aromatic_rings = sum(1 for ring in num_rings if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring))
        
        # Construire la description
        description = []
        
        # Atomes majeurs
        if atoms.get('C', 0) > 0:
            description.append(f"C{atoms['C']}" if atoms['C'] > 1 else "C")
        if atoms.get('H', 0) > 0:
            description.append(f"H{atoms['H']}" if atoms['H'] > 1 else "H")
        if atoms.get('O', 0) > 0:
            description.append(f"O{atoms['O']}" if atoms['O'] > 1 else "O")
        if atoms.get('N', 0) > 0:
            description.append(f"N{atoms['N']}" if atoms['N'] > 1 else "N")
        if atoms.get('S', 0) > 0:
            description.append(f"S{atoms['S']}" if atoms['S'] > 1 else "S")
        if atoms.get('Cl', 0) > 0:
            description.append(f"Cl{atoms['Cl']}" if atoms['Cl'] > 1 else "Cl")
        if atoms.get('F', 0) > 0:
            description.append(f"F{atoms['F']}" if atoms['F'] > 1 else "F")
        if atoms.get('Br', 0) > 0:
            description.append(f"Br{atoms['Br']}" if atoms['Br'] > 1 else "Br")
        
        # Structure
        if num_aromatic_rings > 0:
            description.append(f"{num_aromatic_rings} aromatic ring(s)")
        elif len(num_rings) > 0:
            description.append(f"{len(num_rings)} ring(s)")
        
        # Groupes fonctionnels
        has_oh = any(atom.GetDegree() == 1 and atom.GetSymbol() == 'O' for atom in mol.GetAtoms())
        if has_oh:
            description.append("hydroxyl")
        
        has_nh = any(atom.GetDegree() == 1 and atom.GetSymbol() == 'N' for atom in mol.GetAtoms())
        if has_nh:
            description.append("amine")
        
        has_c_o = any(atom.GetSymbol() == 'C' and any(n.GetSymbol() == 'O' and bond.GetBondType() == Chem.BondType.DOUBLE for n, bond in zip([mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())], [])) for atom in mol.GetAtoms())
        
        return " - ".join(description) if description else "complex organic molecule"
    except Exception:
        return "unknown structure"

def generate():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERREUR : Checkpoint introuvable : {CHECKPOINT_PATH}")
        return

    # Résoudre l'index de la méta-catégorie ciblée
    if isinstance(TARGET_META_CATEGORY, int):
        target_idx = TARGET_META_CATEGORY
        target_name = META_CATEGORY_NAMES[target_idx]
    else:
        target_name = TARGET_META_CATEGORY
        target_idx = META_CATEGORY_NAMES.index(target_name)

    print(f"Méta-catégorie ciblée : {target_name} (index {target_idx}/{len(META_CATEGORY_NAMES)})")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state = checkpoint.get('state_dict', checkpoint)

    real_x_dim = state['model.mlp_in_X.0.weight'].shape[1]
    real_e_dim = state['model.mlp_in_E.0.weight'].shape[1]
    real_y_dim = state['model.mlp_in_y.0.weight'].shape[1]
    out_X = state['model.mlp_out_X.2.weight'].shape[0]
    out_E = state['model.mlp_out_E.2.weight'].shape[0]

    cfg = checkpoint.get('hyper_parameters', {}).get('cfg', checkpoint.get('cfg'))
    OmegaConf.set_struct(cfg, False)
    cfg.model.input_dims = {'X': real_x_dim, 'E': real_e_dim, 'y': real_y_dim}
    cfg.model.output_dims = {'X': out_X, 'E': out_E, 'y': 0}  # Le modèle ne prédit pas y

    # ==============================================================================
    # 3. INITIALISATION DU MODÈLE
    # ==============================================================================
    datamodule = OdorDataModule(cfg)
    dataset_infos = OdorInfos(datamodule=datamodule, cfg=cfg)

    dataset_infos.input_dims = cfg.model.input_dims
    dataset_infos.output_dims = cfg.model.output_dims
    dataset_infos.y_dim = real_y_dim

    # Load training SMILES for novelty computation (same split as OdorDataset)
    train_smiles = None
    target_category_fps = []
    target_category_train_smiles = []
    try:
        csv_path = os.path.join(project_root, datamodule.datadir, "Multi-Labelled_Smiles_Odors_dataset.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_smiles = df["nonStereoSMILES"].astype(str).tolist()
            rng = np.random.RandomState(42)
            perm = rng.permutation(len(all_smiles))
            n_train = int(0.8 * len(all_smiles))
            train_smiles = [all_smiles[i] for i in perm[:n_train]]

            # Build reference fingerprints for the target meta-category (for similarity checks)
            label_cols = [c for c in df.columns if c not in ["nonStereoSMILES", "descriptors"]]
            meta_df = aggregate_to_meta(df, label_cols)
            target_mask = meta_df[target_name] == 1

            for sm in df.loc[target_mask, "nonStereoSMILES"].astype(str).tolist():
                mol = Chem.MolFromSmiles(sm)
                if mol is None:
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                target_category_fps.append(fp)
                target_category_train_smiles.append(sm)
    except Exception:
        train_smiles = None
        target_category_fps = []
        target_category_train_smiles = []

    def find_best_match(smiles):
        """Find the closest training molecule in the target meta-category."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or len(target_category_fps) == 0:
                return None, None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            sims = DataStructs.BulkTanimotoSimilarity(fp, target_category_fps)
            best_idx = int(np.argmax(sims))
            return target_category_train_smiles[best_idx], float(sims[best_idx])
        except Exception:
            return None, None

    extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

    train_metrics = TrainMolecularMetrics(dataset_infos)
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles=train_smiles)
    visualization_tools = MolecularVisualization(remove_h=cfg.dataset.remove_h, dataset_infos=dataset_infos)

    model = DiscreteDenoisingDiffusion.load_from_checkpoint(
        CHECKPOINT_PATH, map_location=device, strict=False,
        dataset_infos=dataset_infos, train_metrics=train_metrics,
        sampling_metrics=sampling_metrics, visualization_tools=visualization_tools,
        extra_features=extra_features, domain_features=domain_features
    )

    model = model.to(device).eval()
    model.print = print
    model.trainer = type('D', (), {'is_global_zero': True, 'current_epoch': 0})()

    # ==============================================================================
    # 4. CONDITIONNEMENT PAR MÉTA-CATÉGORIE
    # ==============================================================================
    original_forward = model.model.forward

    def conditional_forward(X, E, y, node_mask):
        y_cond = torch.zeros_like(y)
        y_cond[:, target_idx] = 1.0
        return original_forward(X, E, y_cond, node_mask)

    model.model.forward = conditional_forward

    # ==============================================================================
    # 5. BOUCLE DE GÉNÉRATION & CALCUL DES PROPRIÉTÉS
    # ==============================================================================
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_filename = os.path.join(RESULTS_DIR, f"generated_{target_name}_with_props.csv")

    with open(csv_filename, "w") as f:
        f.write("SMILES,SMILES_description,descriptors,MolWt,LogP,NumHDonors,NumHAcceptors,TPSA,QED,MolFormula,InChI,InChIKey,closest_train_smiles,closest_train_category,tanimoto_sim,is_close_to_target,is_novel,is_duplicate,valid,validity_reason\n")

    atom_types = dataset_infos.atom_decoder
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

    total_valid = 0
    total_attempts = 0
    duplicate_count = 0
    generated_graphs = []
    seen_smiles = set()

    print(f"\nLancement de la génération ({target_name}) avec calculs RDKit")
    print(f"Objectif : {NUM_BATCHES} x {BATCH_SIZE} = {NUM_BATCHES * BATCH_SIZE} tentatives")

    b = 0
    with torch.no_grad():
        while b < NUM_BATCHES:
            print(f"\n--- Batch {b+1}/{NUM_BATCHES} ---")

            try:
                samples = model.sample_batch(
                    batch_id=0, batch_size=BATCH_SIZE, save_final=True,
                    keep_chain=1, number_chain_steps=50
                )
            except AssertionError:
                print("Graphe dégénéré détecté. Relance du batch...")
                continue
            except Exception as e:
                print(f"Erreur inattendue : {e}. Relance du batch...")
                continue

            batch_data = []
            for _, s in enumerate(samples):
                if hasattr(s, 'X'):
                    X_i = s.X.argmax(dim=-1) if s.X.dim() > 1 else s.X
                    E_i = s.E.argmax(dim=-1) if s.E.dim() > 2 else s.E
                else:
                    X_i = s[0].argmax(dim=-1) if s[0].dim() > 1 else s[0]
                    E_i = s[1].argmax(dim=-1) if s[1].dim() > 2 else s[1]

                total_attempts += 1
                generated_graphs.append((X_i, E_i))

                mol = Chem.RWMol()
                node_map = {}

                for n in range(X_i.shape[0]):
                    atom_idx = X_i[n].item()
                    if atom_idx < len(atom_types):
                        idx = mol.AddAtom(Chem.Atom(atom_types[atom_idx]))
                        node_map[n] = idx

                for u in range(X_i.shape[0]):
                    for v in range(u + 1, X_i.shape[0]):
                        bond_idx = E_i[u, v].item()
                        if 0 < bond_idx <= len(bond_types):
                            if u in node_map and v in node_map:
                                mol.AddBond(node_map[u], node_map[v], bond_types[bond_idx-1])

                try:
                    final_mol = mol.GetMol()
                    sanitize_error = None
                    try:
                        Chem.SanitizeMol(final_mol)
                    except Exception as e:
                        sanitize_error = str(e)

                    sm = None
                    try:
                        sm = Chem.MolToSmiles(final_mol)
                    except Exception:
                        sm = None

                    # Determine validity and reason
                    valid = False
                    reason = "unknown"
                    if sm is None:
                        reason = "no_smiles"
                    elif sanitize_error is not None:
                        reason = "sanitize_failed"
                    elif FILTER_FRAGMENTS and "." in sm:
                        reason = "fragment"
                    else:
                        valid = True
                        reason = "ok"
                        
                        # OPTION 3: Appliquer filtrage chimique post-génération
                        property_ok, property_reason = filter_by_properties(final_mol)
                        if not property_ok:
                            valid = False
                            reason = property_reason

                    closest_smiles, tanimoto = find_best_match(sm) if sm is not None else (None, None)
                    is_close = (tanimoto is not None and tanimoto >= SIMILARITY_THRESHOLD)
                    is_novel = True
                    if NOVELTY_ONLY and tanimoto is not None and NOVELTY_TANIMOTO is not None:
                        is_novel = tanimoto < NOVELTY_TANIMOTO

                    mol_wt = logp = h_donors = h_acceptors = tpsa = qed_score = None
                    formula = None
                    inchi = None
                    inchi_key = None
                    if valid:
                        mol_wt = Descriptors.MolWt(final_mol)
                        logp = Descriptors.MolLogP(final_mol)
                        h_donors = rdMolDescriptors.CalcNumHBD(final_mol)
                        h_acceptors = rdMolDescriptors.CalcNumHBA(final_mol)
                        tpsa = rdMolDescriptors.CalcTPSA(final_mol)
                        qed_score = QED.qed(final_mol)
                        formula = rdMolDescriptors.CalcMolFormula(final_mol)
                        try:
                            inchi = Chem.MolToInchi(final_mol)
                            inchi_key = Chem.InchiToInchiKey(inchi)
                        except Exception:
                            inchi = None
                            inchi_key = None

                    batch_data.append((
                        sm, target_name, mol_wt, logp, h_donors, h_acceptors, tpsa, qed_score,
                        formula, inchi, inchi_key,
                        closest_smiles, target_name if closest_smiles is not None else None,
                        tanimoto, is_close, is_novel, valid, reason
                    ))
                except Exception:
                    # If we hit any unexpected error, still record the attempt as invalid
                    batch_data.append((
                        None, target_name, None, None, None, None, None, None,
                        None, None, None, None, None, None, False, False, False, "exception"
                    ))

            if batch_data:
                with open(csv_filename, "a") as f:
                    for sm, desc, mw, lp, hd, ha, tpsa, qed, formula, inchi, inchi_key, close_sm, close_cat, tanimoto, is_close, is_novel, valid, reason in batch_data:
                        # NE SAUVEGARDER QUE LES MOLÉCULES VALIDES (filtrer les invalides)
                        if not valid:
                            continue
                        
                        is_dup = (sm in seen_smiles) if sm is not None else False
                        if sm is not None:
                            seen_smiles.add(sm)
                        if is_dup:
                            duplicate_count += 1
                        # Format values safely
                        mw_s = f"{mw:.2f}" if mw is not None else ""
                        lp_s = f"{lp:.2f}" if lp is not None else ""
                        tpsa_s = f"{tpsa:.2f}" if tpsa is not None else ""
                        qed_s = f"{qed:.3f}" if qed is not None else ""
                        
                        # Traduire SMILES en description
                        smiles_desc = translate_smiles(sm)

                        f.write(
                            f"{sm},{smiles_desc},{desc},{mw_s},{lp_s},{hd},{ha},{tpsa_s},{qed_s},{formula},{inchi},{inchi_key},{close_sm},{close_cat},{tanimoto},{is_close},{is_novel},{is_dup},{valid},{reason}\n"
                        )

            # item[-2] == valid, item[15] == is_novel
            valid_count = sum(1 for item in batch_data if item[-2] and (not NOVELTY_ONLY or item[15]))
            total_valid += valid_count
            print(f"{valid_count} molécules valides (après filtre nouveauté). (Total: {total_valid})")

            b += 1

    # Compute and save validity / uniqueness / novelty metrics on the full generated set
    os.makedirs(os.path.join(RESULTS_DIR, f"graphs/{target_name}"), exist_ok=True)
    validity_dict, rdkit_metrics, all_smiles = compute_molecular_metrics(generated_graphs, train_smiles, dataset_infos)
    validity, relaxed_validity, uniqueness, novelty = rdkit_metrics[0]

    summary = {
        "target_meta_category": target_name,
        "total_attempts": total_attempts,
        "total_valid": total_valid,
        "unique_saved": len(seen_smiles),
        "duplicate_count": duplicate_count,
        "validity": validity,
        "relaxed_validity": relaxed_validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "dataset_train_smiles_available": train_smiles is not None,
        "filter_fragments": FILTER_FRAGMENTS,
        "similarity_threshold": SIMILARITY_THRESHOLD
    }

    summary_path = os.path.join(RESULTS_DIR, f"generated_{target_name}_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTerminé ! {total_valid} molécules sauvegardées dans '{csv_filename}'")
    print(f"Résumé des métriques sauvegardé dans '{summary_path}'")

if __name__ == "__main__":
    generate()
