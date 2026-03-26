"""
Analysis of generated molecules vs real dataset.
Compares molecular properties and substructures
between generated molecules for a class and real molecules of that class.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
import os
import warnings
warnings.filterwarnings('ignore')
RDLogger.logger().setLevel(RDLogger.ERROR)

# ==============================================================================
# CONFIG
# ==============================================================================
GENERATED_CSV = "results/generated_woody_with_props.csv"
TARGET_CLASS = "woody"
REAL_CSV = "data/odor/Multi-Labelled_Smiles_Odors_dataset.csv"
OUTPUT_DIR = "graphs"

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

FUNCTIONAL_GROUPS = {
    "Hydroxyl (-OH)": "[OX2H]",
    "Aldehyde (-CHO)": "[CX3H1](=O)[#6]",
    "Ketone (C=O)": "[#6][CX3](=O)[#6]",
    "Ester (-COO-)": "[#6][CX3](=O)[OX2H0][#6]",
    "Ether (-O-)": "[OD2]([#6])[#6]",
    "Amine (-NH)": "[NX3;H1,H2;!$(NC=O)]",
    "Thiol (-SH)": "[#16X2H]",
    "Sulfide (-S-)": "[#16X2H0]([#6])[#6]",
    "Aromatic ring": "a",
    "Phenol": "[OX2H][cX3]:[c]",
    "Benzene ring": "c1ccccc1",
    "Alkene (C=C)": "[CX3]=[CX3]",
    "Lactone": "[#6]1~[#6]~[#6]~[OX2]~[CX3](=O)~1",
    "Terpene (isoprene)": "[CH2]=[C]([CH3])[CH2]",
    "Furan": "c1ccoc1",
    "Pyridine": "c1ccncc1",
    "Indole": "c1ccc2[nH]ccc2c1",
    "Methoxy (-OCH3)": "[OX2]([#6])[CH3]",
    "Acetyl (-COCH3)": "[CX3](=O)[CH3]",
}

PROPERTIES = ["MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA"]


def compute_properties(smiles_list):
    """Compute RDKit properties for a list of SMILES."""
    records = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        records.append({
            "SMILES": smi,
            "MolWt": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
            "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        })
    return pd.DataFrame(records)


def detect_fg(smiles_list):
    """Detect functional groups for a list of SMILES."""
    patterns = {}
    for name, smarts in FUNCTIONAL_GROUPS.items():
        pat = Chem.MolFromSmarts(smarts)
        if pat:
            patterns[name] = pat

    results = {name: [] for name in patterns}
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        valid_smiles.append(smi)
        for name, pat in patterns.items():
            results[name].append(1 if mol.HasSubstructMatch(pat) else 0)

    return pd.DataFrame(results), valid_smiles


def main():
    # ==============================================================================
    # 1. LOAD DATA
    # ==============================================================================
    print(f"Analysis of generated molecules: {TARGET_CLASS}")
    print("=" * 60)

    # CSV has unquoted commas in InChI, read SMILES (1st column) directly
    with open(GENERATED_CSV) as f:
        lines = f.readlines()
    gen_smiles = [line.split(",")[0] for line in lines[1:] if line.strip() and not line.startswith("#")]
    gen_smiles = [s for s in gen_smiles if s and s != "SMILES"]
    print(f"Generated molecules: {len(gen_smiles)}")

    # Load real molecules of the same class
    real_df = pd.read_csv(REAL_CSV)
    odor_cols = [c for c in real_df.columns if c not in ("nonStereoSMILES", "descriptors")]
    members = [m for m in META_CATEGORIES[TARGET_CLASS] if m in odor_cols]
    real_df["is_target"] = real_df[members].max(axis=1)
    real_smiles = real_df[real_df["is_target"] == 1]["nonStereoSMILES"].tolist()
    print(f"Real molecules ({TARGET_CLASS}): {len(real_smiles)}")

    # ==============================================================================
    # 2. PROPERTIES
    # ==============================================================================
    print("\nComputing properties...")
    gen_props = compute_properties(gen_smiles)
    real_props = compute_properties(real_smiles)
    gen_props["source"] = "Generated"
    real_props["source"] = "Real"

    print(f"  Valid generated: {len(gen_props)}")
    print(f"  Valid real: {len(real_props)}")

    # ==============================================================================
    # 3. COMPARATIVE STATS
    # ==============================================================================
    print(f"\n{'Property':<20s} {'Generated (mean±std)':<22s} {'Real (mean±std)':<22s}")
    print("-" * 64)
    for prop in PROPERTIES:
        gm, gs = gen_props[prop].mean(), gen_props[prop].std()
        rm, rs = real_props[prop].mean(), real_props[prop].std()
        print(f"{prop:<20s} {gm:>7.2f} ± {gs:<7.2f}    {rm:>7.2f} ± {rs:<7.2f}")

    # ==============================================================================
    # 4. FIGURE 1: COMPARED DISTRIBUTIONS (VIOLIN PLOTS)
    # ==============================================================================
    combined = pd.concat([gen_props, real_props], ignore_index=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i in range(len(PROPERTIES), len(axes)):
        axes[i].set_visible(False)
    for i, prop in enumerate(PROPERTIES):
        ax = axes[i]
        sns.violinplot(data=combined, x="source", y=prop, ax=ax,
                       palette={"Generated": "#e74c3c", "Real": "#3498db"}, cut=0)
        ax.set_title(prop, fontsize=13, fontweight="bold")
        ax.set_xlabel("")

    plt.suptitle(f"Molecular properties: Generated vs Real ({TARGET_CLASS})",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"compare_properties_{TARGET_CLASS}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")

    # ==============================================================================
    # 5. FIGURE 2: COMPARED SUBSTRUCTURES
    # ==============================================================================
    gen_fg, _ = detect_fg(gen_smiles)
    real_fg, _ = detect_fg(real_smiles)

    gen_freq = gen_fg.mean() * 100
    real_freq = real_fg.mean() * 100

    fg_compare = pd.DataFrame({"Generated": gen_freq, "Real": real_freq})
    # Keep only groups with >2% in at least one
    fg_compare = fg_compare[(fg_compare > 2).any(axis=1)]
    fg_compare = fg_compare.sort_values("Real", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(fg_compare) * 0.4)))
    y_pos = np.arange(len(fg_compare))
    width = 0.35
    ax.barh(y_pos - width/2, fg_compare["Real"], width, label="Real", color="#3498db", alpha=0.8)
    ax.barh(y_pos + width/2, fg_compare["Generated"], width, label="Generated", color="#e74c3c", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fg_compare.index)
    ax.set_xlabel("Frequency (%)")
    ax.set_title(f"Functional groups: Generated vs Real ({TARGET_CLASS})",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"compare_substructures_{TARGET_CLASS}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ==============================================================================
    # 6. FIGURE 3: MolWt & LogP DISTRIBUTION (2D scatter)
    # ==============================================================================
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(real_props["MolWt"], real_props["LogP"], alpha=0.3, s=15,
               c="#3498db", label=f"Real (n={len(real_props)})")
    ax.scatter(gen_props["MolWt"], gen_props["LogP"], alpha=0.6, s=30,
               c="#e74c3c", marker="x", label=f"Generated (n={len(gen_props)})")
    ax.set_xlabel("MolWt (g/mol)", fontsize=12)
    ax.set_ylabel("LogP", fontsize=12)
    ax.set_title(f"Chemical space MolWt vs LogP ({TARGET_CLASS})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"chemical_space_{TARGET_CLASS}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ==============================================================================
    # 7. SUMMARY STATS
    # ==============================================================================
    print(f"\n{'=' * 60}")
    print(f"SUMMARY - {TARGET_CLASS}")
    print(f"{'=' * 60}")
    print(f"Valid generated molecules: {len(gen_props)} / {len(gen_smiles)} ({100*len(gen_props)/len(gen_smiles):.0f}%)")

    # Uniqueness
    unique = len(set(gen_smiles))
    print(f"Unique SMILES: {unique} / {len(gen_smiles)} ({100*unique/len(gen_smiles):.0f}%)")

    # Novelty (not in real dataset)
    real_set = set(real_smiles)
    novel = sum(1 for s in gen_smiles if s not in real_set)
    print(f"Novel molecules: {novel} / {len(gen_smiles)} ({100*novel/len(gen_smiles):.0f}%)")

    # Olfactory volatility criterion: MW < 300 Da and LogP between 1.5 and 5.5
    volatile = ((gen_props["MolWt"] < 300) & (gen_props["LogP"] >= 1.5) & (gen_props["LogP"] <= 5.5)).sum()
    print(f"Olfactory volatiles (MW<300 & 1.5<=LogP<=5.5): {volatile} / {len(gen_props)} ({100*volatile/len(gen_props):.0f}%)")

    # Mean Tanimoto (read from CSV — 6th field from the right)
    tanimoto_vals = []
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.strip().split(",")
        try:
            tanimoto_vals.append(float(fields[-6]))
        except (ValueError, IndexError):
            pass
    if tanimoto_vals:
        print(f"Mean Tanimoto (vs train {TARGET_CLASS}): {np.mean(tanimoto_vals):.3f} ± {np.std(tanimoto_vals):.3f}")

    print(f"\nCharts saved in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
