# src/datasets/odor_dataset.py

import os
import os.path as osp
import shutil
import pathlib

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

from src.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos


# ==============================================================================
# MÉTA-CATÉGORIES D'ODEUR (12 classes au lieu de 138)
# ==============================================================================
META_CATEGORIES = {
    "floral": ["floral", "rose", "jasmin", "lily", "muguet", "violet", "hyacinth",
               "geranium", "lavender", "orangeflower", "chamomile", "hawthorn"],
    "fruity": ["fruity", "apple", "apricot", "banana", "berry", "cherry", "grape",
               "grapefruit", "lemon", "melon", "orange", "peach", "pear", "pineapple",
               "plum", "raspberry", "strawberry", "tropical", "black currant", "fruit skin",
               "juicy", "ripe"],
    "sweet": ["sweet", "vanilla", "caramellic", "honey", "chocolate", "cocoa",
              "coconut", "creamy", "buttery", "milky", "dairy"],
    "woody": ["woody", "cedar", "sandalwood", "pine", "vetiver", "terpenic",
              "balsamic", "cortex", "dry"],
    "green": ["green", "grassy", "herbal", "leafy", "hay", "tea", "fresh",
              "cucumber", "vegetable", "weedy", "natural"],
    "spicy": ["spicy", "cinnamon", "clove", "warm", "pungent", "sharp",
              "cooling", "mint", "camphoreous", "aromatic"],
    "animal_musk": ["animal", "musk", "leathery", "fishy", "sweaty", "meaty",
                    "beefy", "musty"],
    "earthy": ["earthy", "mushroom", "nutty", "hazelnut", "roasted", "coffee",
               "tobacco", "smoky", "popcorn"],
    "citrus": ["citrus", "bergamot", "ozone", "clean", "soapy"],
    "chemical": ["solvent", "ethereal", "metallic", "medicinal", "phenolic",
                 "sulfurous", "gassy", "burnt", "oily", "bitter", "alcoholic"],
    "gourmand": ["almond", "malty", "rummy", "brandy", "cognac", "winey",
                 "cooked", "potato", "savory", "celery", "tomato", "radish",
                 "onion", "garlic", "cabbage", "cheesy", "alliaceous", "fermented",
                 "sour"],
    "powdery_amber": ["amber", "powdery", "anisic", "coumarinic", "orris",
                      "waxy", "aldehydic", "ketonic", "lactonic", "fatty"],
}

# Liste ordonnée des méta-catégories (l'ordre définit les indices dans y)
META_CATEGORY_NAMES = list(META_CATEGORIES.keys())
NUM_META_CATEGORIES = len(META_CATEGORY_NAMES)  # 12


def aggregate_to_meta(df, label_cols):
    """
    Agrège les 138 colonnes binaires en 12 méta-catégories.
    Une méta-catégorie vaut 1 si au moins un de ses descripteurs vaut 1.
    Retourne un DataFrame avec 12 colonnes.
    """
    meta_df = pd.DataFrame(index=df.index)
    for meta_name, members in META_CATEGORIES.items():
        present = [m for m in members if m in label_cols]
        if present:
            meta_df[meta_name] = df[present].max(axis=1).astype(np.float32)
        else:
            meta_df[meta_name] = 0.0
    return meta_df


def smiles_to_pyg(smiles: str, remove_h: bool) -> Data | None:
    """
    Convert SMILES to a PyG graph:
      - x: one-hot atom types
      - edge_index: bidirectional edges
      - edge_attr: one-hot bond type with class 0 reserved for "no bond"
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if not remove_h:
        mol = Chem.AddHs(mol)

    # Skip degenerate molecules (no edges)
    if mol.GetNumAtoms() < 2 or mol.GetNumBonds() == 0:
        return None

    if remove_h:
        atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3}
    else:
        atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}

    bond_encoder = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    # Node features
    type_idx = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in atom_encoder:
            return None
        type_idx.append(atom_encoder[sym])

    num_atom_types = len(atom_encoder)
    x = F.one_hot(torch.tensor(type_idx, dtype=torch.long), num_classes=num_atom_types).float()

    # Edges
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        if bt not in bond_encoder:
            return None
        t = bond_encoder[bt] + 1  # 0 reserved for "no bond"
        row += [a, b]
        col += [b, a]
        edge_type += [t, t]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bond_encoder) + 1).float()  # 5 classes

    # Stable edge ordering
    N = mol.GetNumAtoms()
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class OdorDataset(InMemoryDataset):
    """
    Odor dataset from CSV with 12 méta-catégories d'odeur.
      - SMILES column: nonStereoSMILES
      - 138 binary columns → agrégées en 12 méta-catégories
    """
    def __init__(self, stage: str, root: str, remove_h: bool,
                 transform=None, pre_transform=None, pre_filter=None):
        self.stage = stage
        self.remove_h = remove_h

        if self.stage == "train":
            self.file_idx = 0
        elif self.stage == "val":
            self.file_idx = 1
        elif self.stage == "test":
            self.file_idx = 2
        else:
            raise ValueError(f"Unknown stage={stage} (expected train/val/test)")

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx], weights_only=False)

    @property
    def raw_file_names(self):
        return ["Multi-Labelled_Smiles_Odors_dataset.csv"]

    @property
    def processed_file_names(self):
        tag = "no_h" if self.remove_h else "h"
        return [f"proc_tr_{tag}_meta.pt", f"proc_val_{tag}_meta.pt", f"proc_test_{tag}_meta.pt"]

    def download(self):
        src = osp.join(self.root, self.raw_file_names[0])
        dst = self.raw_paths[0]
        os.makedirs(self.raw_dir, exist_ok=True)

        if osp.exists(dst):
            return

        if osp.exists(src):
            shutil.copyfile(src, dst)
            return

        raise FileNotFoundError(f"CSV not found. Put it at {src} (or directly at {dst}).")

    def process(self):
        RDLogger.DisableLog("rdApp.*")

        df = pd.read_csv(self.raw_paths[0])

        if "nonStereoSMILES" not in df.columns:
            raise ValueError("CSV is missing column 'nonStereoSMILES'.")

        # Colonnes de labels originales (138)
        label_cols = [c for c in df.columns if c not in ["nonStereoSMILES", "descriptors"]]
        df[label_cols] = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)

        # Agrégation en 12 méta-catégories
        meta_df = aggregate_to_meta(df, label_cols)
        print(f"[OdorDataset] Agrégation : {len(label_cols)} descripteurs → {NUM_META_CATEGORIES} méta-catégories")
        print(f"[OdorDataset] Méta-catégories : {META_CATEGORY_NAMES}")
        print(f"[OdorDataset] Distribution :")
        for col in META_CATEGORY_NAMES:
            count = int(meta_df[col].sum())
            print(f"  {col:<18s} {count:5d} ({100*count/len(df):.1f}%)")

        # Deterministic split 80/10/10
        n = len(df)
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        splits = [
            perm[:n_train],
            perm[n_train:n_train + n_val],
            perm[n_train + n_val:],
        ]

        for split_idx, indices in enumerate(splits):
            data_list = []
            for i in indices:
                row = df.iloc[int(i)]
                smiles = row["nonStereoSMILES"]

                pyg = smiles_to_pyg(smiles, remove_h=self.remove_h)
                if pyg is None:
                    continue

                # 12 méta-catégories au lieu de 138 descripteurs
                labels = meta_df.iloc[int(i)].values.astype(np.float32)
                y = torch.from_numpy(labels).unsqueeze(0)  # (1, 12)

                pyg.y = y
                pyg.idx = int(i)

                if self.pre_filter is not None and not self.pre_filter(pyg):
                    continue
                if self.pre_transform is not None:
                    pyg = self.pre_transform(pyg)

                data_list.append(pyg)

            torch.save(self.collate(data_list), self.processed_paths[split_idx])
        print(f"[OdorDataset] Processing terminé. y.shape = (1, {NUM_META_CATEGORIES})")


class OdorDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        self.remove_h = cfg.dataset.remove_h

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(str(base_path), self.datadir)

        datasets = {
            "train": OdorDataset(stage="train", root=root_path, remove_h=self.remove_h),
            "val": OdorDataset(stage="val", root=root_path, remove_h=self.remove_h),
            "test": OdorDataset(stage="test", root=root_path, remove_h=self.remove_h),
        }
        super().__init__(cfg, datasets)


class OdorInfos(AbstractDatasetInfos):
    """Infos pour DiGress avec 12 méta-catégories d'odeur."""
    def __init__(self, datamodule, cfg, recompute_statistics: bool = False):
        self.name = "odor"
        self.need_to_strip = False
        self.remove_h = bool(cfg.dataset.remove_h)

        self.max_n_nodes = 64

        # Bond types
        self.bond_encoder = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        self.bond_decoder = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]

        # Atoms + valencies + weights
        if self.remove_h:
            self.atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3}
            self.atom_decoder = ['C', 'N', 'O', 'F']
            self.num_atom_types = 4
            self.valencies = torch.tensor([4, 3, 2, 1], dtype=torch.long)
            self.atom_weights = {'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998}
        else:
            self.atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
            self.atom_decoder = ['H', 'C', 'N', 'O', 'F']
            self.num_atom_types = 5
            self.valencies = torch.tensor([1, 4, 3, 2, 1], dtype=torch.long)
            self.atom_weights = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998}

        self.max_weight = float(max(self.atom_weights.values()) * self.max_n_nodes)

        # Distributions (uniform safe defaults)
        self.n_nodes = torch.ones(self.max_n_nodes + 1)
        self.node_types = torch.ones(self.num_atom_types) / self.num_atom_types
        self.edge_types = torch.ones(len(self.bond_encoder) + 1) / (len(self.bond_encoder) + 1)

        max_valency = int(self.valencies.max().item())
        self.valency_distribution = torch.ones(max_valency + 1)
        self.valency_distribution = self.valency_distribution / self.valency_distribution.sum()

        max_degree = 8
        self.degree_distribution = torch.ones(max_degree + 1)
        self.degree_distribution = self.degree_distribution / self.degree_distribution.sum()

        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

        # y_dim = 12 méta-catégories + 13 extra features (graph features + time)
        raw_y_dim = datamodule.train_dataloader().dataset[0].y.shape[-1]  # 12
        self.y_dim = raw_y_dim + 13  # 25 total


if __name__ == "__main__":
    root = "/home/khalil/DiGress/data/odor"
    ds = OdorDataset(stage="train", root=root, remove_h=True)
    print("len(train) =", len(ds))
    print("y.shape =", ds[0].y.shape)
    print("META_CATEGORY_NAMES =", META_CATEGORY_NAMES)
