"""
Generates a 3x3 grid of the 9 best generated molecules by Tanimoto similarity.
Each molecule is annotated with MW and Tanimoto similarity.
"""

import os
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
RDLogger.logger().setLevel(RDLogger.ERROR)

# ==============================================================================
# CONFIG
# ==============================================================================
GENERATED_CSV = "results/generated_floral_with_props.csv"
TARGET_CLASS  = "floral"
OUTPUT_DIR    = f"graphs/{TARGET_CLASS}_molecules"
TOP_N         = 9
MOL_IMG_SIZE  = (400, 320)


# ==============================================================================
# READ CSV
# ==============================================================================
def load_molecules(csv_path):
    """Read SMILES, MW (col 3) and Tanimoto (6th field from the right)."""
    mols = []
    with open(csv_path) as f:
        f.readline()  # skip header
        for line in f:
            if not line.strip():
                continue
            fields = line.strip().split(",")
            smi = fields[0]
            try:
                mw       = float(fields[3])
                tanimoto = float(fields[-6])
            except (ValueError, IndexError):
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mols.append((mol, smi, mw, tanimoto))
    return mols


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_mols = load_molecules(GENERATED_CSV)
    print(f"Valid molecules loaded: {len(all_mols)}")

    # Sort by descending Tanimoto -> top-9
    top = sorted(all_mols, key=lambda x: x[3], reverse=True)[:TOP_N]

    mols   = [m for m, *_ in top]
    labels = [f"MW={mw:.0f} Da  |  Tanimoto={tan:.3f}"
              for _, _, mw, tan in top]

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=3,
        subImgSize=MOL_IMG_SIZE,
        legends=labels,
        returnPNG=False,
    )

    out_path = os.path.join(OUTPUT_DIR, f"top9_tanimoto_{TARGET_CLASS}.png")
    img.save(out_path)
    print(f"Saved: {out_path}")

    for i, (_, smi, mw, tan) in enumerate(top, 1):
        print(f"  {i}. Tanimoto={tan:.3f}  MW={mw:.0f}  {smi}")


if __name__ == "__main__":
    main()
