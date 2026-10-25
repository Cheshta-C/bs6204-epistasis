"""Audit ProteinGym for double-mutant data.

Setup (run once, on your machine or Colab):
  1. Download the ProteinGym substitution benchmark from https://proteingym.org/download
     (file: DMS_ProteinGym_substitutions.zip) and unzip it.
  2. Also download the reference file DMS_substitutions.csv from the same page.
  3. pip install pandas
Usage:
  python audit_proteingym.py /path/to/DMS_ProteinGym_substitutions

Output: audit_proteingym.csv, one row per assay, sorted by number of double mutants.
"""
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
rows = []
for f in sorted(root.glob("*.csv")):
    if f.name == "DMS_substitutions.csv":
        continue
    df = pd.read_csv(f)
    if "mutant" not in df.columns:
        continue
    # ProteinGym encodes multiple mutations as 'A12G:L40P' (colon-separated)
    n_mut = df["mutant"].astype(str).str.count(":") + 1
    rows.append({
        "assay": f.stem,
        "total": len(df),
        "singles": int((n_mut == 1).sum()),
        "doubles": int((n_mut == 2).sum()),
        "triples_plus": int((n_mut >= 3).sum()),
    })

out = pd.DataFrame(rows).sort_values("doubles", ascending=False)
# Usable for epistasis training: enough doubles AND enough singles to compute the additive term
out["usable"] = (out["doubles"] >= 500) & (out["singles"] >= 50)
out.to_csv("audit_proteingym.csv", index=False)

print(out.head(20).to_string(index=False))
print(f"\nAssays scanned: {len(out)}")
print(f"Assays with any doubles: {(out['doubles'] > 0).sum()}")
print(f"Usable assays (>=500 doubles, >=50 singles): {out['usable'].sum()}")
print(f"Total doubles across usable assays: {out.loc[out['usable'], 'doubles'].sum()}")
