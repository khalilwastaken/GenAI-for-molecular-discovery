#!/usr/bin/env python
"""
Analyze frequent molecular substructures/functional groups by odor meta-category.
Uses RDKit SMARTS matching to detect functional groups in SMILES molecules,
then computes frequency statistics and generates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit import RDLogger
import os
import warnings
warnings.filterwarnings('ignore')
RDLogger.logger().setLevel(RDLogger.ERROR)

# === Configuration ===
DATA_PATH = "data/odor/Multi-Labelled_Smiles_Odors_dataset.csv"
OUTPUT_DIR = "graphs"

META_CATEGORIES = {
    "floral": ["floral", "rose", "jasmin", "lily", "muguet", "violet", "hyacinth", "geranium", "lavender", "orangeflower", "chamomile", "hawthorn"],
    "fruity": ["fruity", "apple", "apricot", "banana", "berry", "cherry", "grape", "grapefruit", "lemon", "melon", "orange", "peach", "pear", "pineapple", "plum", "raspberry", "strawberry", "tropical", "black currant", "fruit skin", "juicy", "ripe"],
    "sweet": ["sweet", "vanilla", "caramellic", "honey", "chocolate", "cocoa", "coconut", "creamy", "buttery", "milky", "dairy"],
    "woody": ["woody", "cedar", "sandalwood", "pine", "vetiver", "terpenic", "balsamic", "cortex", "dry"],
    "green": ["green", "grassy", "herbal", "leafy", "hay", "tea", "fresh", "cucumber", "vegetable", "weedy", "natural"],
    "spicy": ["spicy", "cinnamon", "clove", "warm", "pungent", "sharp", "cooling", "mint", "camphoreous", "aromatic"],
    "animal_musk": ["animal", "musk", "leathery", "fishy", "sweaty", "meaty", "beefy", "musty"],
    "earthy": ["earthy", "mushroom", "nutty", "hazelnut", "roasted", "coffee", "tobacco", "smoky", "popcorn"],
    "citrus": ["citrus", "bergamot", "ozone", "clean", "soapy"],
    "chemical": ["solvent", "ethereal", "metallic", "medicinal", "phenolic", "sulfurous", "gassy", "burnt", "oily", "bitter", "alcoholic"],
    "gourmand": ["almond", "malty", "rummy", "brandy", "cognac", "winey", "cooked", "potato", "savory", "celery", "tomato", "radish", "onion", "garlic", "cabbage", "cheesy", "alliaceous", "fermented", "sour"],
    "powdery_amber": ["amber", "powdery", "anisic", "coumarinic", "orris", "waxy", "aldehydic", "ketonic", "lactonic", "fatty"],
}

# ~28 important functional groups as SMARTS patterns
FUNCTIONAL_GROUPS = {
    "Hydroxyl (-OH)": "[OX2H]",
    "Aldehyde (-CHO)": "[CX3H1](=O)[#6]",
    "Ketone (C=O)": "[#6][CX3](=O)[#6]",
    "Carboxylic acid (-COOH)": "[CX3](=O)[OX2H1]",
    "Ester (-COO-)": "[#6][CX3](=O)[OX2H0][#6]",
    "Ether (-O-)": "[OD2]([#6])[#6]",
    "Primary amine (-NH2)": "[NX3;H2;!$(NC=O)]",
    "Secondary amine (-NHR)": "[NX3;H1;!$(NC=O)]([#6])[#6]",
    "Tertiary amine (-NR3)": "[NX3;H0;!$(NC=O)]([#6])([#6])[#6]",
    "Amide (-CONH-)": "[NX3][CX3](=[OX1])[#6]",
    "Thiol (-SH)": "[#16X2H]",
    "Sulfide (-S-)": "[#16X2H0]([#6])[#6]",
    "Nitrile (-CN)": "[NX1]#[CX2]",
    "Aromatic ring": "a",
    "Phenol": "[OX2H][cX3]:[c]",
    "Benzene ring": "c1ccccc1",
    "Alkene (C=C)": "[CX3]=[CX3]",
    "Alkyne (C#C)": "[CX2]#[CX2]",
    "Lactone (cyclic ester)": "[#6]1~[#6]~[#6]~[OX2]~[CX3](=O)~1",
    "Terpene (isoprene unit)": "[CH2]=[C]([CH3])[CH2]",
    "Epoxide": "C1OC1",
    "Furan": "c1ccoc1",
    "Pyridine": "c1ccncc1",
    "Indole": "c1ccc2[nH]ccc2c1",
    "Methoxy (-OCH3)": "[OX2]([#6])[CH3]",
    "Acetyl (-COCH3)": "[CX3](=O)[CH3]",
    "Nitro (-NO2)": "[NX3](=O)=O",
    "Halide": "[F,Cl,Br,I]",
}


def load_and_prepare_data():
    """Load dataset and compute meta-category labels."""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Number of molecules: {len(df)}")

    # Get binary odor columns (all except SMILES and descriptors)
    odor_cols = [c for c in df.columns if c not in ("nonStereoSMILES", "descriptors")]
    print(f"  Number of odor label columns: {len(odor_cols)}")

    # Compute meta-categories
    for meta, labels in META_CATEGORIES.items():
        matching = [l for l in labels if l in odor_cols]
        df[f"meta_{meta}"] = df[matching].max(axis=1) if matching else 0
        print(f"  Meta-category '{meta}': {len(matching)} labels matched, "
              f"{df[f'meta_{meta}'].sum()} molecules")

    return df, odor_cols


def detect_functional_groups(df):
    """For each molecule, detect presence of each functional group."""
    print("\nParsing SMILES and detecting functional groups...")
    smarts_compiled = {}
    for name, smarts in FUNCTIONAL_GROUPS.items():
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            print(f"  WARNING: Could not parse SMARTS for '{name}': {smarts}")
        else:
            smarts_compiled[name] = pat

    results = {name: [] for name in smarts_compiled}
    valid_mask = []

    for i, smi in enumerate(df["nonStereoSMILES"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_mask.append(False)
            for name in smarts_compiled:
                results[name].append(0)
            continue
        valid_mask.append(True)
        for name, pat in smarts_compiled.items():
            has_match = 1 if mol.HasSubstructMatch(pat) else 0
            results[name].append(has_match)

    fg_df = pd.DataFrame(results, index=df.index)
    n_valid = sum(valid_mask)
    print(f"  Valid molecules: {n_valid}/{len(df)} ({100*n_valid/len(df):.1f}%)")
    print(f"  Functional groups detected: {len(smarts_compiled)}")

    return fg_df, valid_mask


def compute_frequencies(df, fg_df, valid_mask):
    """Compute functional group frequency (%) per meta-category."""
    print("\nComputing frequencies per meta-category...")
    meta_cols = [f"meta_{m}" for m in META_CATEGORIES]
    meta_names = list(META_CATEGORIES.keys())
    fg_names = list(fg_df.columns)

    # Filter to valid molecules
    valid_idx = df.index[valid_mask]
    fg_valid = fg_df.loc[valid_idx]
    df_valid = df.loc[valid_idx]

    # Overall frequency
    overall_freq = fg_valid.mean() * 100

    # Per meta-category frequency
    freq_table = pd.DataFrame(index=fg_names, columns=meta_names, dtype=float)
    counts_table = pd.DataFrame(index=fg_names, columns=meta_names, dtype=float)

    for meta in meta_names:
        mask = df_valid[f"meta_{meta}"] == 1
        n_cat = mask.sum()
        if n_cat > 0:
            freq_table[meta] = fg_valid.loc[mask].mean() * 100
            counts_table[meta] = fg_valid.loc[mask].sum()
        else:
            freq_table[meta] = 0.0
            counts_table[meta] = 0

    freq_table["overall"] = overall_freq.values
    print(f"  Frequency table shape: {freq_table.shape}")

    return freq_table, overall_freq


def plot_heatmap(freq_table):
    """Plot heatmap of functional group frequencies by meta-category."""
    print("\nGenerating heatmap...")
    meta_names = list(META_CATEGORIES.keys())
    plot_data = freq_table[meta_names].copy()

    # Filter: groups with >5% in at least one category OR enriched >1.5x in any category
    freq_mask = (plot_data > 5).any(axis=1)
    overall = freq_table["overall"]
    enrichment = plot_data.div(overall.replace(0, np.nan), axis=0)
    enrich_mask = (enrichment > 1.5).any(axis=1)
    mask = freq_mask | enrich_mask
    plot_data = plot_data.loc[mask]
    print(f"  Groups shown (>5% or enriched >1.5x): {len(plot_data)}")

    # Sort by mean frequency
    plot_data = plot_data.loc[plot_data.mean(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(14, max(8, len(plot_data) * 0.45)))
    sns.heatmap(
        plot_data,
        annot=True, fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Frequency (%)"},
        ax=ax,
    )
    ax.set_title("Functional Group Frequency (%) by Odor Meta-Category", fontsize=14, fontweight='bold')
    ax.set_xlabel("Meta-Category", fontsize=12)
    ax.set_ylabel("Functional Group", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "substructure_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_enrichment_subplots(freq_table, overall_freq):
    """For each meta-category, bar chart of top 10 enriched functional groups."""
    print("\nGenerating enrichment bar charts...")
    meta_names = list(META_CATEGORIES.keys())

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    axes = axes.flatten()

    for idx, meta in enumerate(meta_names):
        ax = axes[idx]
        cat_freq = freq_table[meta]
        # Enrichment = frequency in category / frequency in full dataset
        # Avoid division by zero
        enrichment = cat_freq / overall_freq.replace(0, np.nan)
        enrichment = enrichment.dropna().sort_values(ascending=False).head(10)

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(enrichment)))
        bars = ax.barh(range(len(enrichment)), enrichment.values, color=colors)
        ax.set_yticks(range(len(enrichment)))
        ax.set_yticklabels(enrichment.index, fontsize=8)
        ax.set_xlabel("Enrichment ratio", fontsize=9)
        ax.set_title(meta.replace("_", " ").title(), fontsize=11, fontweight='bold')
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.invert_yaxis()

    plt.suptitle("Top 10 Enriched Functional Groups per Meta-Category\n(Enrichment = category freq / overall freq)",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "substructure_enrichment_by_category.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_discriminative_groups(freq_table):
    """Grouped bar chart of top 5 most discriminative functional groups."""
    print("\nGenerating discriminative groups chart...")
    meta_names = list(META_CATEGORIES.keys())
    plot_data = freq_table[meta_names].copy()

    # Variance across categories
    variance = plot_data.var(axis=1).sort_values(ascending=False)
    top5 = variance.head(5).index.tolist()
    print(f"  Top 5 most discriminative groups: {top5}")

    subset = plot_data.loc[top5].T  # meta-categories as rows, groups as columns

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(meta_names))
    width = 0.15
    colors = plt.cm.Set2(np.linspace(0, 1, 5))

    for i, grp in enumerate(top5):
        offset = (i - 2) * width
        ax.bar(x + offset, subset[grp].values, width, label=grp, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in meta_names], rotation=45, ha='right')
    ax.set_ylabel("Frequency (%)", fontsize=12)
    ax.set_title("Top 5 Most Discriminative Functional Groups Across Meta-Categories\n(Highest variance in frequency)",
                 fontsize=13, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "substructure_discriminative_groups.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("Molecular Substructure Analysis by Odor Meta-Category")
    print("=" * 60)

    # Step 1: Load data
    df, odor_cols = load_and_prepare_data()

    # Step 2-3: Detect functional groups
    fg_df, valid_mask = detect_functional_groups(df)

    # Step 4: Compute frequencies
    freq_table, overall_freq = compute_frequencies(df, fg_df, valid_mask)

    # Step 5a: Heatmap
    plot_heatmap(freq_table)

    # Step 5b: Enrichment subplots
    plot_enrichment_subplots(freq_table, overall_freq)

    # Step 5c: Discriminative groups
    plot_discriminative_groups(freq_table)

    # Step 6: Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "substructure_frequencies_by_metacategory.csv")
    freq_table.to_csv(csv_path, float_format="%.2f")
    print(f"\nSaved CSV: {csv_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    meta_names = list(META_CATEGORIES.keys())
    print("\nOverall functional group frequencies (top 10):")
    top_overall = overall_freq.sort_values(ascending=False).head(10)
    for name, val in top_overall.items():
        print(f"  {name:30s} {val:6.1f}%")

    print("\nMost distinctive group per meta-category:")
    for meta in meta_names:
        enrichment = freq_table[meta] / overall_freq.replace(0, np.nan)
        enrichment = enrichment.dropna()
        if len(enrichment) > 0:
            best = enrichment.idxmax()
            print(f"  {meta:20s} -> {best} (enrichment: {enrichment[best]:.2f}x)")

    print("\nDone! All outputs saved to graphs/")


if __name__ == "__main__":
    main()
