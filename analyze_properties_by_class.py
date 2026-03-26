"""
Analyze molecular properties by odor meta-category.

Loads the odor dataset, computes RDKit molecular properties, groups molecules
by meta-category, and generates statistical summaries and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import warnings
warnings.filterwarnings("ignore")

# ── Meta-categories mapping ──────────────────────────────────────────────────
META_CATEGORIES = {
    "floral": ["floral", "rose", "jasmin", "lily", "muguet", "violet", "hyacinth",
                "geranium", "lavender", "orangeflower", "chamomile", "hawthorn"],
    "fruity": ["fruity", "apple", "apricot", "banana", "berry", "cherry", "grape",
               "grapefruit", "lemon", "melon", "orange", "peach", "pear", "pineapple",
               "plum", "raspberry", "strawberry", "tropical", "black currant",
               "fruit skin", "juicy", "ripe"],
    "sweet": ["sweet", "vanilla", "caramellic", "honey", "chocolate", "cocoa",
              "coconut", "creamy", "buttery", "milky", "dairy"],
    "woody": ["woody", "cedar", "sandalwood", "pine", "vetiver", "terpenic",
              "balsamic", "cortex", "dry"],
    "green": ["green", "grassy", "herbal", "leafy", "hay", "tea", "fresh",
              "cucumber", "vegetable", "weedy", "natural"],
    "spicy": ["spicy", "cinnamon", "clove", "warm", "pungent", "sharp", "cooling",
              "mint", "camphoreous", "aromatic"],
    "animal_musk": ["animal", "musk", "leathery", "fishy", "sweaty", "meaty",
                    "beefy", "musty"],
    "earthy": ["earthy", "mushroom", "nutty", "hazelnut", "roasted", "coffee",
               "tobacco", "smoky", "popcorn"],
    "citrus": ["citrus", "bergamot", "ozone", "clean", "soapy"],
    "chemical": ["solvent", "ethereal", "metallic", "medicinal", "phenolic",
                 "sulfurous", "gassy", "burnt", "oily", "bitter", "alcoholic"],
    "gourmand": ["almond", "malty", "rummy", "brandy", "cognac", "winey", "cooked",
                 "potato", "savory", "celery", "tomato", "radish", "onion", "garlic",
                 "cabbage", "cheesy", "alliaceous", "fermented", "sour"],
    "powdery_amber": ["amber", "powdery", "anisic", "coumarinic", "orris", "waxy",
                      "aldehydic", "ketonic", "lactonic", "fatty"],
}

PROPERTIES = ["MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA",
              "NumRotatableBonds", "NumAromaticRings", "FractionCSP3"]

DATA_PATH = "data/odor/Multi-Labelled_Smiles_Odors_dataset.csv"
OUT_DIR = "graphs"


def compute_properties(smiles: str) -> dict:
    """Compute RDKit molecular properties for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {p: np.nan for p in PROPERTIES}
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
    }


def load_and_prepare():
    """Load CSV, compute meta-category labels and molecular properties."""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} molecules, {df.shape[1]} columns")

    # Identify available binary label columns (everything except SMILES and descriptors)
    label_cols = [c for c in df.columns if c not in ("nonStereoSMILES", "descriptors")]

    # Build meta-category binary columns
    for cat, members in META_CATEGORIES.items():
        present = [m for m in members if m in label_cols]
        if present:
            df[f"meta_{cat}"] = df[present].max(axis=1)
        else:
            df[f"meta_{cat}"] = 0
        print(f"  {cat}: {df[f'meta_{cat}'].sum():>5d} molecules  ({len(present)} / {len(members)} labels found)")

    # Compute RDKit properties
    print("Computing RDKit properties (this may take a minute)...")
    props = df["nonStereoSMILES"].apply(compute_properties).apply(pd.Series)
    df = pd.concat([df, props], axis=1)
    n_valid = df["MolWt"].notna().sum()
    print(f"  Valid molecules: {n_valid} / {len(df)}")

    return df, label_cols


def compute_stats(df):
    """Compute mean/std/median of each property per meta-category."""
    rows = []
    for cat in META_CATEGORIES:
        mask = df[f"meta_{cat}"] == 1
        sub = df.loc[mask, PROPERTIES]
        for prop in PROPERTIES:
            vals = sub[prop].dropna()
            rows.append({
                "meta_category": cat,
                "property": prop,
                "count": int(mask.sum()),
                "mean": vals.mean(),
                "std": vals.std(),
                "median": vals.median(),
            })
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(f"{OUT_DIR}/property_stats_by_metacategory.csv", index=False)
    print(f"Saved stats table to {OUT_DIR}/property_stats_by_metacategory.csv")
    return stats_df


def plot_boxplots(df):
    """Box plots of each property by meta-category (subplots grid)."""
    cats = list(META_CATEGORIES.keys())
    meta_col = "meta_category_label"

    # Build long-form dataframe: one row per molecule per category it belongs to
    records = []
    for cat in cats:
        mask = df[f"meta_{cat}"] == 1
        sub = df.loc[mask, PROPERTIES].copy()
        sub[meta_col] = cat
        records.append(sub)
    long = pd.concat(records, ignore_index=True)

    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
    axes = axes.flatten()
    for i, prop in enumerate(PROPERTIES):
        ax = axes[i]
        sns.boxplot(data=long, x=meta_col, y=prop, ax=ax, showfliers=False,
                    palette="Set3", order=cats)
        ax.set_title(prop, fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("")
    plt.suptitle("Molecular Properties by Odor Meta-Category", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/boxplots_properties_by_metacategory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved boxplots to {OUT_DIR}/boxplots_properties_by_metacategory.png")


def plot_radar(stats_df):
    """Radar/spider chart of normalized mean properties per category."""
    cats = list(META_CATEGORIES.keys())
    pivot = stats_df.pivot(index="meta_category", columns="property", values="mean")
    pivot = pivot.loc[cats, PROPERTIES]

    # Normalize each property to [0, 1]
    normed = (pivot - pivot.min()) / (pivot.max() - pivot.min() + 1e-12)

    angles = np.linspace(0, 2 * np.pi, len(PROPERTIES), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    cmap = plt.cm.get_cmap("tab20", len(cats))

    for i, cat in enumerate(cats):
        values = normed.loc[cat].tolist() + [normed.loc[cat].iloc[0]]
        ax.plot(angles, values, linewidth=1.8, label=cat, color=cmap(i))
        ax.fill(angles, values, alpha=0.06, color=cmap(i))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(PROPERTIES, fontsize=9)
    ax.set_title("Normalized Mean Properties by Meta-Category", fontsize=14,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/radar_properties_by_metacategory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved radar chart to {OUT_DIR}/radar_properties_by_metacategory.png")


def plot_heatmap(stats_df):
    """Heatmap of z-score normalized mean property values."""
    cats = list(META_CATEGORIES.keys())
    pivot = stats_df.pivot(index="meta_category", columns="property", values="mean")
    pivot = pivot.loc[cats, PROPERTIES]

    # Z-score normalize each column
    z = (pivot - pivot.mean()) / (pivot.std() + 1e-12)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(z, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "z-score"})
    ax.set_title("Z-Score Normalized Mean Properties by Meta-Category",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/heatmap_properties_by_metacategory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {OUT_DIR}/heatmap_properties_by_metacategory.png")


def plot_significance(df):
    """
    Mann-Whitney U test: for each category x property, test whether molecules
    in the category differ from those not in the category. Produce annotated heatmap.
    """
    cats = list(META_CATEGORIES.keys())
    pval_matrix = pd.DataFrame(index=cats, columns=PROPERTIES, dtype=float)

    for cat in cats:
        mask = df[f"meta_{cat}"] == 1
        for prop in PROPERTIES:
            in_vals = df.loc[mask, prop].dropna()
            out_vals = df.loc[~mask, prop].dropna()
            if len(in_vals) < 5 or len(out_vals) < 5:
                pval_matrix.loc[cat, prop] = 1.0
            else:
                _, p = mannwhitneyu(in_vals, out_vals, alternative="two-sided")
                pval_matrix.loc[cat, prop] = p

    pval_matrix = pval_matrix.astype(float)

    # Star annotation
    def stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""

    annot = pval_matrix.applymap(stars)
    log_p = -np.log10(pval_matrix.clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(log_p, annot=annot, fmt="s", cmap="YlOrRd",
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "-log10(p-value)"})
    ax.set_title("Statistical Significance (Mann-Whitney U) of Property Differences\n"
                 "* p<0.05  ** p<0.01  *** p<0.001",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/significance_properties_by_metacategory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved significance heatmap to {OUT_DIR}/significance_properties_by_metacategory.png")

    # Print summary of significant results
    n_sig = (pval_matrix < 0.05).sum().sum()
    n_total = pval_matrix.size
    print(f"  {n_sig} / {n_total} category-property pairs significant at p<0.05")


def main():
    df, label_cols = load_and_prepare()
    stats_df = compute_stats(df)
    print("\n--- Generating plots ---")
    plot_boxplots(df)
    plot_radar(stats_df)
    plot_heatmap(stats_df)
    plot_significance(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
