"""
Analysis of the 138 odor classes in the dataset.
Produces:
  - Frequency of each descriptor
  - Co-occurrence matrix
  - Proposed meta-categories
  - Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ==============================================================================
# 1. LOADING
# ==============================================================================
CSV_PATH = "data/odor/Multi-Labelled_Smiles_Odors_dataset.csv"
df = pd.read_csv(CSV_PATH)

label_cols = [c for c in df.columns if c not in ("nonStereoSMILES", "descriptors")]
labels = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

print(f"Dataset: {len(df)} molecules, {len(label_cols)} odor descriptors\n")

# ==============================================================================
# 2. FREQUENCY OF EACH DESCRIPTOR
# ==============================================================================
freq = labels.sum().sort_values(ascending=False)

print("=" * 60)
print("TOP 30 most frequent descriptors:")
print("=" * 60)
for i, (name, count) in enumerate(freq.head(30).items()):
    pct = 100 * count / len(df)
    bar = "█" * int(pct / 2)
    print(f"  {i+1:2d}. {name:<18s} {count:5d}  ({pct:5.1f}%)  {bar}")

print(f"\n  ... {(freq == 0).sum()} descriptors with 0 examples")
print(f"  ... {(freq < 50).sum()} descriptors with < 50 examples")
print(f"  ... {(freq >= 50).sum()} descriptors with >= 50 examples")

# ==============================================================================
# 3. DISTRIBUTION OF LABELS PER MOLECULE
# ==============================================================================
labels_per_mol = labels.sum(axis=1)
print(f"\nLabels per molecule: min={labels_per_mol.min()}, max={labels_per_mol.max()}, "
      f"mean={labels_per_mol.mean():.1f}, median={labels_per_mol.median():.0f}")

# ==============================================================================
# 4. MATRICE DE CO-OCCURRENCE (top 30)
# ==============================================================================
top30 = freq.head(30).index.tolist()
labels_top30 = labels[top30]
cooccurrence = labels_top30.T.dot(labels_top30)

# Normalize by min(freq_i, freq_j) for a Jaccard-like score
freq_top30 = labels_top30.sum()
norm = np.minimum.outer(freq_top30.values, freq_top30.values)
norm[norm == 0] = 1
cooccurrence_norm = cooccurrence / norm

# ==============================================================================
# 5. VISUALISATIONS
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 5a. Frequency barplot (top 30)
ax = axes[0, 0]
freq.head(30).plot.barh(ax=ax, color="steelblue")
ax.set_xlabel("Number of molecules")
ax.set_title("Top 30 odor descriptors (frequency)")
ax.invert_yaxis()

# 5b. Distribution of labels per molecule
ax = axes[0, 1]
labels_per_mol.hist(bins=range(0, labels_per_mol.max() + 2), ax=ax, color="coral", edgecolor="white")
ax.set_xlabel("Number of descriptors per molecule")
ax.set_ylabel("Number of molecules")
ax.set_title("Multi-label distribution")

# 5c. Normalized co-occurrence heatmap (top 30)
ax = axes[1, 0]
mask = np.triu(np.ones_like(cooccurrence_norm, dtype=bool), k=1)
sns.heatmap(cooccurrence_norm, mask=mask, cmap="YlOrRd", ax=ax,
            xticklabels=True, yticklabels=True, vmin=0, vmax=1,
            cbar_kws={"label": "Normalized co-occurrence"})
ax.set_title("Normalized co-occurrence (top 30)")
ax.tick_params(axis='both', labelsize=7)

# 5d. Cumulative frequency (how many classes to cover X% of labels)
ax = axes[1, 1]
cumsum = freq.cumsum() / freq.sum() * 100
ax.plot(range(1, len(cumsum) + 1), cumsum.values, "o-", markersize=2, color="green")
ax.axhline(y=80, color="red", linestyle="--", alpha=0.7, label="80%")
ax.axhline(y=95, color="orange", linestyle="--", alpha=0.7, label="95%")
n_80 = (cumsum <= 80).sum() + 1
n_95 = (cumsum <= 95).sum() + 1
ax.axvline(x=n_80, color="red", linestyle=":", alpha=0.5)
ax.axvline(x=n_95, color="orange", linestyle=":", alpha=0.5)
ax.set_xlabel("Number of descriptors (sorted by frequency)")
ax.set_ylabel("Cumulative % of labels")
ax.set_title(f"Cumulative coverage ({n_80} classes = 80%, {n_95} classes = 95%)")
ax.legend()

plt.tight_layout()
plt.savefig("graphs/odor_class_analysis.png", dpi=150)
print(f"\nChart saved: graphs/odor_class_analysis.png")
plt.close()

# ==============================================================================
# 6. PROPOSED META-CATEGORIES
# ==============================================================================
# Based on olfactory chemistry (Fragrance Wheel / Arctander classification)
META_CATEGORIES = {
    "floral": ["floral", "rose", "jasmin", "lily", "muguet", "violet", "hyacinth",
               "geranium", "lavender", "orangeflower", "chamomile", "hawthorn"],
    "fruity": ["fruity", "apple", "apricot", "banana", "berry", "cherry", "grape",
               "grapefruit", "lemon", "melon", "orange", "peach", "pear", "pineapple",
               "plum", "raspberry", "strawberry", "tropical", "black currant", "fruit skin"],
    "sweet": ["sweet", "vanilla", "caramellic", "honey", "chocolate", "cocoa",
              "coconut", "creamy", "buttery", "milky", "dairy"],
    "woody": ["woody", "cedar", "sandalwood", "pine", "vetiver", "terpenic",
              "balsamic", "cortex"],
    "green": ["green", "grassy", "herbal", "leafy", "hay", "tea", "fresh",
              "cucumber", "vegetable", "weedy"],
    "spicy": ["spicy", "cinnamon", "clove", "warm", "pungent", "sharp",
              "cooling", "mint", "camphoreous"],
    "animal_musk": ["animal", "musk", "leathery", "fishy", "sweaty", "meaty",
                    "beefy", "musty"],
    "earthy": ["earthy", "mushroom", "nutty", "hazelnut", "roasted", "coffee",
               "tobacco", "smoky", "popcorn"],
    "citrus": ["citrus", "bergamot", "ozone", "clean", "soapy"],
    "chemical": ["solvent", "ethereal", "metallic", "medicinal", "phenolic",
                 "sulfurous", "gassy", "burnt", "oily"],
    "gourmand": ["almond", "malty", "rummy", "brandy", "cognac", "winey",
                 "cooked", "potato", "savory", "celery", "tomato", "radish",
                 "onion", "garlic", "cabbage", "cheesy"],
    "powdery_amber": ["amber", "powdery", "anisic", "coumarinic", "orris",
                      "waxy", "aldehydic", "ketonic", "lactonic"],
}

print("\n" + "=" * 60)
print("PROPOSED META-CATEGORIES")
print("=" * 60)

uncovered = set(label_cols)
for meta_name, members in META_CATEGORIES.items():
    present = [m for m in members if m in label_cols]
    total = labels[present].max(axis=1).sum() if present else 0
    pct = 100 * total / len(df)
    uncovered -= set(present)
    print(f"\n  {meta_name.upper()} ({len(present)} descriptors, {int(total)} molecules, {pct:.1f}%)")
    for m in present:
        c = labels[m].sum()
        print(f"    - {m:<18s} {c:4d}")

if uncovered:
    print(f"\n  UNCLASSIFIED: {sorted(uncovered)}")

# ==============================================================================
# 7. SAVE SUMMARY
# ==============================================================================
summary = freq.reset_index()
summary.columns = ["descriptor", "count"]
summary["percentage"] = (100 * summary["count"] / len(df)).round(1)
summary.to_csv("graphs/odor_descriptor_frequencies.csv", index=False)
print(f"\nFrequencies saved: graphs/odor_descriptor_frequencies.csv")
