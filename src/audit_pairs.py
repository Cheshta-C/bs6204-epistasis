"""Step 2 audit: how many doubles have both singles measured, how additive is each assay,
and how many distinct proteins are there?

Usage (from the proteingym_audit folder):
  python3 audit_pairs.py DMS_ProteinGym_substitutions DMS_substitutions.csv

Output: audit_pairs.csv (one row per usable assay) plus a printed summary.
"""
import sys
from pathlib import Path

import pandas as pd

data_dir = Path(sys.argv[1])
meta = pd.read_csv(sys.argv[2])
print("Columns in DMS_substitutions.csv:", list(meta.columns), "\n")


def pick(*names):
    for n in names:
        if n in meta.columns:
            return n
    return None


file_col = pick("DMS_filename", "DMS_id")
prot_col = pick("UniProt_ID", "UniProt_id")
type_col = pick("coarse_selection_type", "selection_type")

rows = []
for f in sorted(data_dir.glob("*.csv")):
    df = pd.read_csv(f, usecols=lambda c: c in ("mutant", "DMS_score"))
    if "mutant" not in df.columns:
        continue
    parts = df["mutant"].astype(str).str.split(":")
    n_mut = parts.str.len()
    singles = df[n_mut == 1].set_index("mutant")["DMS_score"]
    doubles = df[n_mut == 2].copy()
    if len(doubles) == 0:
        continue
    d_parts = parts[n_mut == 2]
    doubles["s_a"] = d_parts.str[0].map(singles).values
    doubles["s_b"] = d_parts.str[1].map(singles).values
    complete = doubles.dropna(subset=["s_a", "s_b"])
    if len(complete) >= 50:
        # Rank correlation between the measured double and the sum of its singles.
        # High value = mostly additive = hard baseline; low = more epistasis to learn.
        add_rho = complete["DMS_score"].rank().corr((complete["s_a"] + complete["s_b"]).rank())
    else:
        add_rho = float("nan")
    rows.append({
        "assay": f.stem,
        "singles": len(singles),
        "doubles": len(doubles),
        "doubles_pair_complete": len(complete),
        "additive_spearman": round(add_rho, 3),
    })

out = pd.DataFrame(rows)

if file_col:
    key = meta[file_col].astype(str).str.replace(".csv", "", regex=False)
    lookup = meta.assign(assay=key)[["assay"] + [c for c in (prot_col, type_col) if c]]
    out = out.merge(lookup, on="assay", how="left")

out["usable"] = out["doubles_pair_complete"] >= 500
out["capped_doubles"] = out["doubles_pair_complete"].clip(upper=20000)
out = out.sort_values("doubles_pair_complete", ascending=False)
out.to_csv("audit_pairs.csv", index=False)

use = out[out["usable"]]
print(use.head(15).to_string(index=False))
print(f"\nUsable assays (>=500 pair-complete doubles): {len(use)}")
print(f"Pair-complete doubles in usable assays: {use['doubles_pair_complete'].sum()}")
print(f"After capping each assay at 20,000: {use['capped_doubles'].sum()}")
print(f"Median additive Spearman across usable assays: {use['additive_spearman'].median()}")
if prot_col and prot_col in use.columns:
    print(f"Distinct proteins among usable assays ({prot_col}): {use[prot_col].nunique()}")
if type_col and type_col in use.columns:
    print("\nUsable assays by assay type:")
    print(use[type_col].value_counts().to_string())
