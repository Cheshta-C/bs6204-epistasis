"""Shared evaluation harness for the haplotype-epistasis project.

Every model (M0 additive, M1 sequence, M2 structure) is scored the SAME way here,
using held-out-PROTEIN cross-validation, so comparisons are fair.

Two ways to use it:

  # 1. The M0 additive baseline (no training needed): predict z_ab = z_a + z_b
  python3 eval_harness.py processed/pairs.csv

  # 2. Score a model's predictions. The predictions CSV must have columns
  #    assay, mutant, pred_z_ab   (one row per double mutant in pairs.csv)
  python3 eval_harness.py processed/pairs.csv --pred my_model_preds.csv --name M1_ESM

Metrics (all computed per assay, then averaged = "macro", plus "pooled" over all rows):
  z_ab Spearman   how well the double-mutant effect is ranked          (overall accuracy)
  z_ab RMSE       error on the double-mutant effect
  epi Spearman    how well the EPISTASIS residual is ranked            (the headline number)
  epi AUROC       detecting strongly epistatic pairs (|epistasis| > --epi-thr)

M0's epistasis prediction is 0 for every pair, so its epi Spearman is ~0 and its
epi AUROC is ~0.5. A deep model earns its keep by beating those two floors on
held-out proteins.

Requires: pandas, numpy, scipy, scikit-learn
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def _spear(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return spearmanr(a, b).correlation


def evaluate(df, pred_z, pred_epi, epi_thr):
    """df has columns z_ab, epistasis; pred_z/pred_epi are aligned arrays."""
    d = df.copy()
    d["pred_z"] = pred_z
    d["pred_epi"] = pred_epi

    per_assay = []
    for assay, g in d.groupby("assay"):
        row = {
            "assay": assay,
            "n": len(g),
            "z_spearman": _spear(g["z_ab"], g["pred_z"]),
            "z_rmse": float(np.sqrt(np.mean((g["z_ab"] - g["pred_z"]) ** 2))),
            "epi_spearman": _spear(g["epistasis"], g["pred_epi"]),
        }
        y = (g["epistasis"].abs() > epi_thr).astype(int)
        if y.nunique() == 2:
            row["epi_auroc"] = roc_auc_score(y, g["pred_epi"].abs())
        else:
            row["epi_auroc"] = np.nan
        per_assay.append(row)
    pa = pd.DataFrame(per_assay)

    y_all = (d["epistasis"].abs() > epi_thr).astype(int)
    pooled = {
        "z_spearman": _spear(d["z_ab"], d["pred_z"]),
        "z_rmse": float(np.sqrt(np.mean((d["z_ab"] - d["pred_z"]) ** 2))),
        "epi_spearman": _spear(d["epistasis"], d["pred_epi"]),
        "epi_auroc": roc_auc_score(y_all, d["pred_epi"].abs()) if y_all.nunique() == 2 else np.nan,
    }
    macro = {
        "z_spearman": pa["z_spearman"].mean(),
        "z_rmse": pa["z_rmse"].mean(),
        "epi_spearman": pa["epi_spearman"].mean(),
        "epi_auroc": pa["epi_auroc"].mean(),
    }
    return pa, macro, pooled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pairs_csv")
    p.add_argument("--pred", help="model predictions CSV with assay,mutant,pred_z_ab")
    p.add_argument("--name", default="M0_additive")
    p.add_argument("--epi-thr", type=float, default=1.0,
                   help="|epistasis| above this (in assay z units) counts as 'strongly epistatic'")
    p.add_argument("--out", default="eval_per_assay.csv")
    args = p.parse_args()

    df = pd.read_csv(args.pairs_csv)

    if args.pred:
        pred = pd.read_csv(args.pred)
        merged = df.merge(pred[["assay", "mutant", "pred_z_ab"]], on=["assay", "mutant"], how="left")
        missing = merged["pred_z_ab"].isna().sum()
        if missing:
            print(f"WARNING: {missing} of {len(merged)} pairs have no prediction and are dropped.")
            merged = merged.dropna(subset=["pred_z_ab"])
        pred_z = merged["pred_z_ab"].to_numpy()
        # a model's implied epistasis = its z_ab prediction minus the additive expectation
        pred_epi = (merged["pred_z_ab"] - merged["z_additive_fit"]).to_numpy()
        df = merged
    else:
        # M0 additive baseline: naive sum of single effects; epistasis prediction is zero
        pred_z = (df["z_a"] + df["z_b"]).to_numpy()
        pred_epi = np.zeros(len(df))

    pa, macro, pooled = evaluate(df, pred_z, pred_epi, args.epi_thr)
    pa.to_csv(args.out, index=False)

    def line(d):
        return (f"z_Spearman={d['z_spearman']:.3f}  z_RMSE={d['z_rmse']:.3f}  "
                f"epi_Spearman={d['epi_spearman']:.3f}  epi_AUROC={d['epi_auroc']:.3f}")

    print(f"\n=== {args.name}  (held-out-protein folds already baked into pairs.csv) ===")
    print(f"pairs evaluated: {len(df)}   assays: {df['assay'].nunique()}")
    print(f"MACRO  (mean over assays): {line(macro)}")
    print(f"POOLED (all rows)        : {line(pooled)}")

    if "selection_type" in df.columns:
        print("\nBy selection type (macro epi_Spearman):")
        for t, g in df.assign(pz=pred_z, pe=pred_epi).groupby("selection_type"):
            rows = [_spear(gg["epistasis"], gg["pe"]) for _, gg in g.groupby("assay")]
            val = np.nan if all(np.isnan(r) for r in rows) else np.nanmean(rows)
            shown = "   n/a (no signal)" if np.isnan(val) else f"{val:.3f}"
            print(f"  {t:20s} assays={g['assay'].nunique():2d}  epi_Spearman={shown}")

    print(f"\nPer-assay metrics written to {args.out}")
    if not args.pred:
        print("\nM0 reference floor: epi_Spearman ~ 0 and epi_AUROC ~ 0.5 by construction.")
        print("M1/M2 must beat these on held-out proteins to show they learn interaction.")


if __name__ == "__main__":
    main()
