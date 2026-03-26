# GenAI for Molecular Discovery
### Conditional Olfactory Molecule Generation via Discrete Graph Diffusion

> M2 Intelligence Artificielle & Robotique — CY Cergy Paris Université  
> Author: **Khalil Hamdaoui** | Supervisor: **Guillaume Renton** (ETIS Lab — UMR 8051)  
> March 2026

---

## Overview

This project adapts **DiGress** — a discrete graph diffusion model — to the conditional generation of molecules with target olfactory profiles. Given a fragrance category (e.g. *floral*, *animal\_musk*, *woody*, *citrus*), the model generates novel molecular structures that are chemically valid and structurally close to the target class.

Rather than engineering molecules by hand or screening existing compounds one by one, we use **inverse design**: a generative model learns the underlying chemical distribution and directly proposes candidate structures satisfying the desired properties.

---

## Key Results

| Category | Validity | Uniqueness | Novelty | Tanimoto |
|---|---|---|---|---|
| animal\_musk | 39.0% | 85.9% | 95.0% | 0.431 ± 0.238 |
| floral | 21.8% | 81.2% | 88.6% | 0.355 ± 0.142 |
| woody | 27.7% | 77.0% | 92.7% | 0.379 ± 0.171 |
| citrus | 24.6% | 78.8% | 90.0% | 0.311 ± 0.135 |

- **Novelty > 88%** across all categories — the model genuinely invents new structures
- **Tanimoto > baseline** — one-hot conditioning steers generation toward the correct chemical family
- No architectural modification to DiGress required

---

## Project Structure
```
├── data/
│   └── odor/
│       ├── Multi-Labelled_Smiles_Odors_dataset.csv   # Raw dataset (4,151 molecules, 138 descriptors)
│       └── dataset_with_metacategories.csv           # Processed dataset with 12 meta-categories
├── src/
│   └── datasets/
│       └── odor_dataset.py                           # OdorDataset, OdorInfos, OdorDataModule
├── outputs/                                          # Training checkpoints and logs
├── results/                                          # Generated molecules (CSV + metrics JSON)
├── graphs/                                           # All figures (violin plots, scatter, heatmaps...)
├── analyze_odor_classes.py                           # Dataset analysis and meta-category aggregation
├── generate_odor.py                                  # Conditional molecule generation
├── analyze_generated.py                              # Comparative analysis (generated vs real)
└── README.md
```

---

## Dataset

**Source:** publicly available multi-labeled SMILES database  
**Size:** 4,151 molecules annotated with up to 138 raw olfactory descriptors  
**Split:** 80/10/10 (train/val/test), seed=42 → 3,321 / 415 / 415 molecules

### Meta-Category Aggregation

The 138 raw descriptors are grouped into **12 meta-categories** using:
1. Frequency analysis (top-30 descriptors cover ~80% of annotations)
2. Normalized co-occurrence matrix: $C_{ij} = |D_i \cap D_j| / \min(|D_i|, |D_j|)$
3. Fragrance Wheel classification

| Category | N | Category | N |
|---|---|---|---|
| fruity | 2282 | floral | 1304 |
| green | 2254 | woody | 1057 |
| sweet | 1930 | animal\_musk | 772 |
| chemical | 1552 | citrus | 514 |
| powdery\_amber | 1335 | earthy | 1006 |
| gourmand | 1218 | spicy | 1076 |

Chemical distinctiveness validated by **Mann-Whitney U test** (p < 0.001 on at least one physico-chemical property per category).

---

## Model

**Base model:** DiGress Graph Transformer (Vignac et al., 2022)  
**Parameters:** 4.8M  
**Atoms:** C, N, O, F (hydrogen-free graphs, max 64 nodes)

| Parameter | Value |
|---|---|
| Input dims (X / E / y) | 12 / 5 / 25 |
| Output dims (X / E) | 4 / 5 |
| Hidden dims (dx / de / dy) | 128 / 32 / 128 |
| Attention heads | 8 |
| FFN dims (nodes / edges) | 512 / 128 |
| Optimizer | AdamW, lr=2×10⁻⁴, wd=10⁻¹² |
| Epochs | 50 |
| Batch size | 64 |
| Hardware | NVIDIA RTX 5060 (8 GB VRAM) |
| Training time | ~2h30 |
| Convergence | Epoch 23 (node CE: 0.426→0.297) |

### Conditioning Mechanism

At inference time, the graph-level conditioning vector **y** (dim=25: 12 category flags + 13 global features) is overridden at every denoising step with a one-hot encoding of the target category:

$$\hat{G}^{t-1} = \phi_\theta(G^t, \mathbf{y} = \mathbf{c}^{(k)})$$

**No architectural change required** — this simply replaces the conditioning signal already used internally by DiGress.

---

## Evaluation

### Generation Metrics (MOSES benchmark)

$$\text{Validity} = \frac{|\{m \mid \texttt{sanitize}(m) = \text{OK}\}|}{1280}$$

$$\text{Uniqueness} = \frac{|\{\text{distinct SMILES} \mid m \in \mathcal{G}_\text{valid}\}|}{|\mathcal{G}_\text{valid}|}$$

$$\text{Novelty} = \frac{|\{m \in \mathcal{G}_\text{valid} \mid \text{SMILES}(m) \notin \mathcal{D}_\text{train}\}|}{|\mathcal{G}_\text{valid}|}$$

All three metrics are computed over all **1,280 generation attempts** (invalid molecules included in the denominator for validity; valid molecules as denominator for uniqueness and novelty).

### Tanimoto Similarity

Morgan fingerprints (radius=2, 2048 bits) via RDKit. For each generated molecule, nearest-neighbor Tanimoto against the target class:

$$T(m, \mathcal{D}_k) = \max_{r \in \mathcal{D}_k} \frac{|\text{FP}(m) \cap \text{FP}(r)|}{|\text{FP}(m) \cup \text{FP}(r)|}$$

Computed on valid **and** filtered molecules (50 ≤ MW ≤ 600 Da, -2 ≤ LogP ≤ 6).

### Functional Group Analysis

28 SMARTS patterns detected via `mol.HasSubstructMatch(pattern)`.  
Enrichment ratio: $E_{g,k} = f_{g,k} / f_{g,\text{overall}}$ — values above 1 indicate over-representation in the category.

---

## Installation
```bash
# Clone DiGress
git clone https://github.com/cvignac/DiGress.git
cd DiGress

# Install dependencies
pip install torch torch-geometric rdkit pytorch-lightning
pip install pandas numpy matplotlib seaborn scipy

# Add project files
cp path/to/odor_dataset.py src/datasets/
cp path/to/analyze_odor_classes.py .
cp path/to/generate_odor.py .
cp path/to/analyze_generated.py .
```

---

## Usage

### 1. Build the dataset
```bash
python analyze_odor_classes.py
# Output: data/odor/dataset_with_metacategories.csv
#         graphs/odor_class_analysis.png
```

### 2. Train the model
```bash
python main.py --dataset odor --guidance_target None
# Checkpoint saved to outputs/<date>/<time>-odor/
```

### 3. Generate molecules
```bash
# Edit TARGET_META_CATEGORY in generate_odor.py (e.g. 'floral')
python generate_odor.py
# Output: results/generated_floral_with_props.csv
#         results/generated_floral_metrics.json
```

### 4. Analyze results
```bash
# Edit CATEGORY in analyze_generated.py
python analyze_generated.py
# Output: graphs/compare_properties_floral.png
#         graphs/chemical_space_floral.png
#         graphs/compare_substructures_floral.png
```

---

## Limitations

1. **Minimal conditioning** — the one-hot override does not modify the diffusion process itself. A more principled approach would incorporate the class signal into the transition matrices (Graph DiT / classifier-free guidance).

2. **MW bias** — the model pulls generated MW toward the global dataset average regardless of the target category. Categories lighter than average (animal\_musk) are overestimated (+22 Da); heavier ones (woody, floral) are underestimated (-35 Da, -24 Da). Explicit MW conditioning via AdaLN would correct this.

3. **Chemical validity ≠ olfactory quality** — whether generated molecules actually smell like musk or floral requires perceptual evaluation or a dedicated odor prediction model.

---

## References

- Vignac et al. (2022). *DiGress: Discrete Denoising Diffusion for Graph Generation.* arXiv:2209.14734
- Liu et al. (2024). *Graph Diffusion Transformers for Multi-Conditional Molecular Generation.* NeurIPS 37
- Polykovskiy et al. (2020). *Molecular Sets (MOSES): A Benchmarking Platform.* Frontiers in Pharmacology
- Bilodeau et al. (2022). *Generative Models for Molecular Discovery.* WIREs Computational Molecular Science
- Landrum, G. RDKit: Open-source cheminformatics. https://www.rdkit.org

---

## License

This project is part of an M2 research project at CY Cergy Paris Université / ETIS Laboratory.  
Base model: DiGress (MIT License).
