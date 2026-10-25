"""Build the clean double-mutant pairs table that every model in the project trains on.

For each usable ProteinGym assay this script:
  1. keeps doubles whose two single mutants were both measured ("pair-complete"),
  2. z-scores DMS scores within the assay so assays are comparable,
  3. fits a robust line  z_AB ~ a + b * (z_A + z_B)  per assay and stores the residual
     as the epistasis label (plus the naive label z_AB - z_A - z_B for comparison),
  4. caps each assay at --cap doubles (random but reproducible) so GB1 does not dominate,
  5. assigns cross-validation folds grouped by protein (UniProt_ID), so the same protein
     never appears in both train and test,
  6. checks that each mutation's wild-type residue matches the reference sequence.

Usage:
  python3 build_pairs.py DMS_ProteinGym_substitutions DMS_substitutions.csv
  python3 build_pairs.py DMS_ProteinGym_substitutions DMS_substitutions.csv --stability-only

Outputs (written to --out, default ./processed):
  pairs.csv        one row per double mutant (the training table)
  sequences.csv    one row per assay: protein ID and wild-type sequence
  assay_fits.csv   per-assay normalisation and additive-fit parameters
  folds.csv        which protein is in which fold

Requires: pandas, numpy
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

p = argparse.ArgumentParser()
p.add_argument("data_dir")
p.add_argument("meta_csv")
p.add_argument("--out", default="processed")
p.add_argument("--min-doubles", type=int, default=500)
p.add_argument("--cap", type=int, default=20000)
p.add_argument("--folds", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--stability-only", action="store_true")
args = p.parse_args()

out_dir = Path(args.out)
out_dir.mkdir(exist_ok=True)
rng = np.random.default_rng(args.seed)

meta = pd.read_csv(args.meta_csv)
meta["assay"] = meta["DMS_filename"].astype(str).str.replace(".csv", "", regex=False)
meta = meta.set_index("assay")


def huber_fit(x, y, delta=1.345, iters=50):
    """Robust straight-line fit y ~ a + b*x (Huber IRLS). Returns a, b."""
    X = np.column_stack([np.ones_like(x), x])
    w = np.ones_like(y)
    beta = np.zeros(2)
    for _ in range(iters):
        W = np.sqrt(w)
        beta_new, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
        r = y - X @ beta_new
        s = np.median(np.abs(r - np.median(r))) / 0.6745 or 1.0
        u = np.abs(r / s)
        w = np.where(u <= delta, 1.0, delta / u)
        if np.allclose(beta, beta_new, atol=1e-8):
            beta = beta_new
            break
        beta = beta_new
    return float(beta[0]), float(beta[1])


def parse(m):
    g = MUT_RE.match(m)
    return (g.group(1), int(g.group(2)), g.group(3)) if g else (None, None, None)


all_pairs, fits, seqs = [], [], []
wt_mismatch = 0

for f in sorted(Path(args.data_dir).glob("*.csv")):
    assay = f.stem
    if assay not in meta.index:
        continue
    info = meta.loc[assay]
    if args.stability_only and info.get("coarse_selection_type") != "Stability":
        continue

    df = pd.read_csv(f, usecols=lambda c: c in ("mutant", "DMS_score"))
    if "mutant" not in df.columns:
        continue
    df = df.dropna(subset=["DMS_score"])
    df["mutant"] = df["mutant"].astype(str)
    n_mut = df["mutant"].str.count(":") + 1

    # z-score within the assay (singles + doubles together)
    mu, sd = df["DMS_score"].mean(), df["DMS_score"].std()
    if not sd or np.isnan(sd):
        continue
    df["z"] = (df["DMS_score"] - mu) / sd

    singles = df[n_mut == 1].drop_duplicates("mutant").set_index("mutant")["z"]
    d = df[n_mut == 2].copy()
    if len(d) < args.min_doubles:        # nothing to do for this assay
        continue
    split = d["mutant"].str.split(":", expand=True)
    if split.shape[1] < 2:               # no properly colon-separated doubles
        continue
    d["mut_a"], d["mut_b"] = split[0], split[1]
    d["z_a"] = d["mut_a"].map(singles)
    d["z_b"] = d["mut_b"].map(singles)
    d = d.dropna(subset=["z_a", "z_b"])
    if len(d) < args.min_doubles:
        continue

    # fit the additive expectation on ALL pair-complete doubles, before capping
    x = (d["z_a"] + d["z_b"]).to_numpy()
    y = d["z"].to_numpy()
    a, b = huber_fit(x, y)
    d["z_additive_fit"] = a + b * x
    d["epistasis"] = y - d["z_additive_fit"]          # main label
    d["epistasis_naive"] = y - x                     # z_AB - z_A - z_B, for comparison

    # parse positions and check wild-type residues against the reference sequence
    seq = str(info["target_seq"])
    for side in ("a", "b"):
        parsed = d[f"mut_{side}"].map(parse)
        d[f"wt_{side}"] = parsed.str[0]
        d[f"pos_{side}"] = parsed.str[1]
        d[f"aa_{side}"] = parsed.str[2]
    d = d.dropna(subset=["pos_a", "pos_b"])
    d["pos_a"] = d["pos_a"].astype(int)
    d["pos_b"] = d["pos_b"].astype(int)
    ok = d.apply(
        lambda r: 0 < r.pos_a <= len(seq) and 0 < r.pos_b <= len(seq)
        and seq[r.pos_a - 1] == r.wt_a and seq[r.pos_b - 1] == r.wt_b,
        axis=1,
    )
    wt_mismatch += int((~ok).sum())
    d = d[ok]
    d["seq_distance"] = (d["pos_a"] - d["pos_b"]).abs()

    n_complete = len(d)
    if n_complete > args.cap:
        d = d.iloc[rng.choice(n_complete, args.cap, replace=False)]

    d["assay"] = assay
    d["UniProt_ID"] = info["UniProt_ID"]
    d["selection_type"] = info.get("coarse_selection_type")
    all_pairs.append(d.rename(columns={"z": "z_ab", "DMS_score": "dms_score_ab"}))
    fits.append({"assay": assay, "UniProt_ID": info["UniProt_ID"],
                 "selection_type": info.get("coarse_selection_type"),
                 "score_mean": mu, "score_sd": sd, "fit_intercept": a, "fit_slope": b,
                 "pair_complete_doubles": n_complete, "kept_doubles": len(d)})
    seqs.append({"assay": assay, "UniProt_ID": info["UniProt_ID"],
                 "seq_len": len(seq), "target_seq": seq})
    print(f"{assay:45s} kept {len(d):6d} / {n_complete:6d}  slope={b:.2f}")

pairs = pd.concat(all_pairs, ignore_index=True)

# protein-grouped folds: greedy balance by row count, deterministic
counts = pairs.groupby("UniProt_ID").size().sort_values(ascending=False)
fold_rows = [0] * args.folds
fold_of = {}
for prot, n in counts.items():
    k = int(np.argmin(fold_rows))
    fold_of[prot] = k
    fold_rows[k] += n
pairs["fold"] = pairs["UniProt_ID"].map(fold_of)

cols = ["assay", "UniProt_ID", "selection_type", "fold", "mutant",
        "mut_a", "wt_a", "pos_a", "aa_a", "mut_b", "wt_b", "pos_b", "aa_b", "seq_distance",
        "z_a", "z_b", "z_ab", "z_additive_fit", "epistasis", "epistasis_naive", "dms_score_ab"]
pairs[cols].to_csv(out_dir / "pairs.csv", index=False)
pd.DataFrame(seqs).to_csv(out_dir / "sequences.csv", index=False)
pd.DataFrame(fits).to_csv(out_dir / "assay_fits.csv", index=False)
pd.Series(fold_of, name="fold").rename_axis("UniProt_ID").reset_index().to_csv(
    out_dir / "folds.csv", index=False)

print("\n=== Summary ===")
print(f"Assays kept: {pairs['assay'].nunique()}   Proteins: {pairs['UniProt_ID'].nunique()}")
print(f"Double mutants in pairs.csv: {len(pairs)}")
print(f"Rows dropped for wild-type mismatch: {wt_mismatch}")
print("Rows per fold:", pairs.groupby("fold").size().to_dict())
print("Proteins per fold:", pairs.groupby("fold")["UniProt_ID"].nunique().to_dict())
print(f"Epistasis label: mean {pairs['epistasis'].mean():.3f}, sd {pairs['epistasis'].std():.3f}")
print(f"Share of pairs with |epistasis| > 1 (assay sd units): {(pairs['epistasis'].abs() > 1).mean():.1%}")
print(f"\nWrote: {', '.join(str(out_dir / n) for n in ['pairs.csv', 'sequences.csv', 'assay_fits.csv', 'folds.csv'])}")
