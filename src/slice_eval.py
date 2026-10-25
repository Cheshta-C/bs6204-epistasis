"""Sliced comparison of M1 (sequence) vs M2 (structure) by pair geometry.

Structure should help ONLY where it can: pairs far apart in sequence but close in 3D.
Averaged over all pairs that lift is diluted. This splits pairs into four slices
(sequence near/far x 3D contact/distant) and reports, per slice, the epistasis-ranking
(Spearman of predicted vs measured epistasis) for M1 and M2, and the gap M2 - M1.

Usage:
  python3 slice_eval.py pairs_struct.csv m1_650_preds.csv m2_preds.csv
Outputs: a printed table, slice_compare.csv, and slice_compare.png (the figure).

Requires: pandas, numpy, scipy  (matplotlib for the figure)
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sp(a, b):
    if len(a) < 20 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return spearmanr(a, b).correlation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_struct"); ap.add_argument("m1_pred"); ap.add_argument("m2_pred")
    ap.add_argument("--seq-far", type=int, default=5)
    ap.add_argument("--contact", type=float, default=8.0)
    args = ap.parse_args()

    p = pd.read_csv(args.pairs_struct)
    m1 = pd.read_csv(args.m1_pred).rename(columns={"pred_z_ab": "m1"})[["assay", "mutant", "m1"]]
    m2 = pd.read_csv(args.m2_pred).rename(columns={"pred_z_ab": "m2"})[["assay", "mutant", "m2"]]
    df = p.merge(m1, on=["assay", "mutant"]).merge(m2, on=["assay", "mutant"])
    df = df.dropna(subset=["m1", "m2", "dist_3d"])
    # each model's implied epistasis = prediction minus the additive expectation
    df["m1_epi"] = df["m1"] - df["z_additive_fit"]
    df["m2_epi"] = df["m2"] - df["z_additive_fit"]
    df["seqfar"] = df["seq_distance"] >= args.seq_far
    df["contact"] = df["dist_3d"] <= args.contact

    rows = []
    for (sf, c), g in df.groupby(["seqfar", "contact"]):
        rows.append({
            "slice": f"{'seq-FAR' if sf else 'seq-near'} / {'3D-contact' if c else '3D-distant'}",
            "n": len(g),
            "M1_epiSpearman": sp(g["epistasis"], g["m1_epi"]),
            "M2_epiSpearman": sp(g["epistasis"], g["m2_epi"]),
        })
    rows.append({"slice": "ALL pairs", "n": len(df),
                 "M1_epiSpearman": sp(df["epistasis"], df["m1_epi"]),
                 "M2_epiSpearman": sp(df["epistasis"], df["m2_epi"])})
    res = pd.DataFrame(rows)
    res["M2_minus_M1"] = res["M2_epiSpearman"] - res["M1_epiSpearman"]
    res.to_csv("slice_compare.csv", index=False)

    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    key = res[res["slice"] == "seq-FAR / 3D-contact"]
    if len(key):
        k = key.iloc[0]
        print(f"\nKey slice (seq-FAR / 3D-contact), where structure SHOULD help:")
        print(f"  M1 {k['M1_epiSpearman']:.3f}  ->  M2 {k['M2_epiSpearman']:.3f}   "
              f"(gain {k['M2_minus_M1']:+.3f}, n={int(k['n']):,})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = ["seq-near / 3D-distant", "seq-near / 3D-contact",
                 "seq-FAR / 3D-distant", "seq-FAR / 3D-contact", "ALL pairs"]
        r = res.set_index("slice").reindex([o for o in order if o in set(res["slice"])])
        x = np.arange(len(r)); w = 0.38
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - w / 2, r["M1_epiSpearman"], w, label="M1 (sequence)", color="#4C78A8")
        ax.bar(x + w / 2, r["M2_epiSpearman"], w, label="M2 (structure)", color="#E45756")
        ax.set_xticks(x); ax.set_xticklabels(r.index, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("epistasis Spearman (held-out proteins)")
        ax.set_title("Structure helps where it should: far in sequence, close in 3D")
        ax.legend(frameon=False); ax.axhline(0, color="#888", lw=0.6)
        for i, (a, b) in enumerate(zip(r["M1_epiSpearman"], r["M2_epiSpearman"])):
            if not np.isnan(b):
                ax.text(x[i] + w / 2, b + 0.005, f"{b:.2f}", ha="center", fontsize=8)
        fig.tight_layout(); fig.savefig("slice_compare.png", dpi=140)
        print("\nWrote slice_compare.csv and slice_compare.png")
    except ImportError:
        print("\nWrote slice_compare.csv (matplotlib missing, skipped the figure)")


if __name__ == "__main__":
    main()
