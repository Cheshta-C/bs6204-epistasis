"""Confidence intervals for the M1 vs M2 comparison, by resampling proteins.

Each model can be given several prediction files (one per training seed). For every
bootstrap resample of proteins (proteins drawn with replacement, all their pairs kept),
each metric is computed for every file, averaged over that model's seeds, and the
difference M2 - M1 is recorded. Proteins, not pairs, are resampled because pairs from
the same protein are not independent.

Metrics (epistasis, held-out proteins):
  macro epistasis Spearman  (mean over assays)
  macro epistasis AUROC     (|epistasis| > 1 vs rest, mean over assays)
  pooled epistasis Spearman, overall and in the four geometry slices
    (sequence near/far at 5 residues x 3D contact/distant at 8 A)

Usage:
  python3 compare_ci.py pairs_struct.csv \
      --m1 m1reg_s0.csv m1reg_s1.csv m1reg_s2.csv \
      --m2 m2_preds.csv m2_s1.csv m2_s2.csv --boot 500 --out ci_m2_vs_m1.tsv
Prediction files: assay, mutant, pred_z_ab (as written by train_m1.py / train_m2.py).
Requires: pandas, numpy, scipy, scikit-learn
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def spearman(x, y):
    if len(x) < 20:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return np.corrcoef(rx, ry)[0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_struct")
    ap.add_argument("--m1", nargs="+", required=True)
    ap.add_argument("--m2", nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seq-far", type=int, default=5)
    ap.add_argument("--contact", type=float, default=8.0)
    ap.add_argument("--epi-thr", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ci_m2_vs_m1.tsv")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # assays with no strongly epistatic pair give empty AUROC means

    df = pd.read_csv(a.pairs_struct)
    files = [("M1", f) for f in a.m1] + [("M2", f) for f in a.m2]
    cols = []
    for k, (_, f) in enumerate(files):
        p = pd.read_csv(f)[["assay", "mutant", "pred_z_ab"]].rename(columns={"pred_z_ab": f"p{k}"})
        df = df.merge(p, on=["assay", "mutant"], how="inner")
        cols.append(f"p{k}")
    df = df.dropna(subset=cols + ["dist_3d"]).reset_index(drop=True)
    print(f"pairs scored by every file: {len(df):,} in {df['UniProt_ID'].nunique()} proteins")

    y = df["epistasis"].to_numpy()
    E = np.stack([(df[c] - df["z_additive_fit"]).to_numpy() for c in cols])  # implied epistasis per file
    model_of = np.array([m for m, _ in files])
    far = (df["seq_distance"] >= a.seq_far).to_numpy()
    con = (df["dist_3d"] <= a.contact).to_numpy()
    slices = {"ALL pairs": np.ones(len(df), bool),
              "seq-near / 3D-distant": ~far & ~con, "seq-near / 3D-contact": ~far & con,
              "seq-FAR / 3D-distant": far & ~con, "seq-FAR / 3D-contact": far & con}

    # per-assay metrics computed once (macro metrics are averages of these)
    assays = df["assay"].to_numpy()
    a_ids = {s: i for i, s in enumerate(pd.unique(assays))}
    a_idx = np.array([a_ids[s] for s in assays])
    sp_a = np.full((len(cols), len(a_ids)), np.nan)
    au_a = np.full((len(cols), len(a_ids)), np.nan)
    for s, i in a_ids.items():
        m = a_idx == i
        lab = (np.abs(y[m]) > a.epi_thr).astype(int)
        for k in range(len(cols)):
            sp_a[k, i] = spearman(y[m], E[k, m])
            if lab.min() != lab.max():
                au_a[k, i] = roc_auc_score(lab, np.abs(E[k, m]))
    prot_of_assay = df.groupby("assay")["UniProt_ID"].first()
    prots = df["UniProt_ID"].unique()
    rows_of = {p: np.where(df["UniProt_ID"].to_numpy() == p)[0] for p in prots}
    assays_of = {p: [a_ids[s] for s in prot_of_assay[prot_of_assay == p].index] for p in prots}

    def metrics(sample):
        idx = np.concatenate([rows_of[p] for p in sample])
        aid = np.concatenate([assays_of[p] for p in sample])
        out = {}
        for k in range(len(cols)):
            out[("macro epistasis Spearman", k)] = np.nanmean(sp_a[k, aid])
            out[("macro epistasis AUROC", k)] = np.nanmean(au_a[k, aid])
            for name, mask in slices.items():
                sel = idx[mask[idx]]
                out[(f"pooled Spearman: {name}", k)] = spearman(y[sel], E[k, sel])
        return out

    names = ["macro epistasis Spearman", "macro epistasis AUROC"] + [f"pooled Spearman: {n}" for n in slices]

    def by_model(m):
        res = {}
        for n in names:
            v1 = np.nanmean([m[(n, k)] for k in range(len(cols)) if model_of[k] == "M1"])
            v2 = np.nanmean([m[(n, k)] for k in range(len(cols)) if model_of[k] == "M2"])
            res[n] = (v1, v2)
        return res

    point = metrics(prots)
    base = by_model(point)
    rng = np.random.default_rng(a.seed)
    diffs = {n: [] for n in names}
    for b in range(a.boot):
        r = by_model(metrics(rng.choice(prots, len(prots), replace=True)))
        for n in names:
            diffs[n].append(r[n][1] - r[n][0])
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b + 1}/{a.boot}")

    rows = []
    for n in names:
        d = np.array(diffs[n])
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        seeds1 = [point[(n, k)] for k in range(len(cols)) if model_of[k] == "M1"]
        seeds2 = [point[(n, k)] for k in range(len(cols)) if model_of[k] == "M2"]
        rows.append({"metric": n, "M1": base[n][0], "M2": base[n][1], "M2_minus_M1": base[n][1] - base[n][0],
                     "ci_low": lo, "ci_high": hi, "share_boot_M2_better": (d > 0).mean(),
                     "M1_seed_range": f"{min(seeds1):.3f}-{max(seeds1):.3f}",
                     "M2_seed_range": f"{min(seeds2):.3f}-{max(seeds2):.3f}"})
    res = pd.DataFrame(rows)
    res.to_csv(a.out, sep="\t", index=False)
    print(f"\nM1 files: {len(a.m1)}, M2 files: {len(a.m2)}; {a.boot} protein resamples")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nA difference is clear only if its 95% CI excludes 0. Seed ranges show training-run variation.")
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
