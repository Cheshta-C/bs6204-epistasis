"""Compute 3D structure features for the haplotype-epistasis project (input to M2).

For each protein's structure (ProteinGym DMS Structures, PDB format), this:
  1. parses heavy-atom coordinates per residue (no external deps -- plain PDB ATOM lines),
  2. aligns the structure to the assay's target_seq and reports the match rate,
  3. computes, for every double-mutant pair in pairs.csv, the minimum heavy-atom
     distance between the two mutated residues  ->  dist_3d  (adds it to pairs),
  4. saves per-protein 3D contact graphs (residue pairs within --contact A) for M2,
  5. runs the de-risking check: among sequence-FAR pairs, do 3D-near pairs show more
     |epistasis| than 3D-far pairs?  (If not, M2's structure channel won't help.)

Usage:
  python3 structure_features.py pairs.csv sequences.csv structures/ \
      --out-pairs pairs_3d.csv --out-contacts contacts.npz
  # structures/ holds one PDB per assay; the script matches files to assays by name.

Requires: numpy, pandas  (matplotlib optional, for the one plot)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U",
}


def parse_pdb_residues(path):
    """Return ordered list of (resname1, Nx3 heavy-atom coords) for the first chain."""
    residues, cur_key, cur_atoms, cur_name, chain0 = [], None, [], None, None
    with open(path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break  # first model only
            if not line.startswith(("ATOM", "HETATM")):
                continue
            element = line[76:78].strip()
            alt = line[16].strip()
            if element == "H" or (alt and alt not in ("A", "1")):
                continue
            resname = line[17:20].strip()
            if resname not in THREE_TO_ONE:
                continue
            chain = line[21]
            if chain0 is None:
                chain0 = chain
            if chain != chain0:
                continue
            key = (chain, line[22:27])  # resseq + icode
            if key != cur_key:
                if cur_key is not None and cur_atoms:
                    residues.append((THREE_TO_ONE[cur_name], np.array(cur_atoms)))
                cur_key, cur_name, cur_atoms = key, resname, []
            try:
                cur_atoms.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                pass
    if cur_key is not None and cur_atoms:
        residues.append((THREE_TO_ONE[cur_name], np.array(cur_atoms)))
    return residues


def best_offset(struct_seq, target_seq):
    """Find integer offset o so struct residue i maps to target index i+o; max matches."""
    best_o, best_hits = 0, -1
    for o in range(-len(target_seq), len(target_seq)):
        hits = sum(1 for i, a in enumerate(struct_seq)
                   if 0 <= i + o < len(target_seq) and target_seq[i + o] == a)
        if hits > best_hits:
            best_hits, best_o = hits, o
    return best_o, best_hits


def residue_min_dist(coords_i, coords_j):
    """Minimum heavy-atom distance between two residues' coordinate sets."""
    diff = coords_i[:, None, :] - coords_j[None, :, :]
    return float(np.sqrt((diff ** 2).sum(-1)).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_csv")
    ap.add_argument("sequences_csv")
    ap.add_argument("structures_dir")
    ap.add_argument("--contact", type=float, default=8.0, help="contact threshold (Angstrom)")
    ap.add_argument("--seq-far", type=int, default=5, help="|pos_a-pos_b| >= this is 'sequence-far'")
    ap.add_argument("--out-pairs", default="pairs_3d.csv")
    ap.add_argument("--out-contacts", default="contacts.npz")
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs_csv)
    seqs = pd.read_csv(args.sequences_csv).set_index("assay")
    files = {os.path.basename(p): p for p in glob.glob(os.path.join(args.structures_dir, "*"))}

    assay_to_uni = dict(zip(pairs["assay"], pairs["UniProt_ID"]))

    # match a structure file to an assay. ProteinGym AF2 files are usually named by
    # UniProt_ID (e.g. SPG1_STRSG.pdb); fall back to the assay name, then its stem.
    def find_file(assay):
        uni = str(assay_to_uni.get(assay, ""))
        for key in (uni, assay):          # most specific first
            if not key:
                continue
            for name, path in files.items():
                if key in name:
                    return path
        return None

    pos_coords = {}      # assay -> {target_pos(1-based): coords}
    contacts = {}        # assay -> (E,2) int array of contacting residue positions (1-based)
    report = []
    for assay in pairs["assay"].unique():
        f = find_file(assay)
        if f is None or assay not in seqs.index:
            report.append((assay, "no structure", 0.0))
            continue
        res = parse_pdb_residues(f)
        struct_seq = "".join(a for a, _ in res)
        target = str(seqs.loc[assay, "target_seq"])
        o, hits = best_offset(struct_seq, target)
        match = hits / max(1, len(struct_seq))
        report.append((assay, os.path.basename(f), match))
        if match < 0.8:
            continue  # alignment too poor to trust
        # map modeled residues to target positions
        cmap = {}
        for i, (_, coords) in enumerate(res):
            tp = i + o + 1  # 1-based target position
            if 1 <= tp <= len(target):
                cmap[tp] = coords
        pos_coords[assay] = cmap
        # contact graph (min heavy-atom distance), sequence-separated residues only
        positions = sorted(cmap)
        edges = []
        for ii in range(len(positions)):
            for jj in range(ii + 1, len(positions)):
                a, b = positions[ii], positions[jj]
                if abs(a - b) < 2:
                    continue
                if residue_min_dist(cmap[a], cmap[b]) <= args.contact:
                    edges.append((a, b))
        contacts[assay] = np.array(edges, dtype=int) if edges else np.empty((0, 2), int)

    # per-pair 3D distance
    d3 = np.full(len(pairs), np.nan)
    for idx, r in pairs.iterrows():
        cm = pos_coords.get(r["assay"])
        if cm is None:
            continue
        ca, cb = cm.get(int(r["pos_a"])), cm.get(int(r["pos_b"]))
        if ca is not None and cb is not None:
            d3[idx] = residue_min_dist(ca, cb)
    pairs["dist_3d"] = d3
    pairs.to_csv(args.out_pairs, index=False)
    if contacts:
        np.savez_compressed(args.out_contacts, **{k: v for k, v in contacts.items()})

    # ---- report ----
    rep = pd.DataFrame(report, columns=["assay", "file", "match_rate"])
    ok = rep[rep["match_rate"] >= 0.8]
    print(f"Structures matched for {len(ok)}/{len(rep)} assays (match_rate >= 0.80).")
    poor = rep[(rep["match_rate"] < 0.8)]
    if len(poor):
        print("  Low/again-check:", ", ".join(f"{a}({m:.2f})" for a, _, m in poor.itertuples(index=False)))

    have = pairs.dropna(subset=["dist_3d"]).copy()
    print(f"\n3D distance computed for {len(have):,}/{len(pairs):,} pairs.")
    if len(have) == 0:
        print("No 3D distances -- check structure files and naming.")
        return

    have["abs_epi"] = have["epistasis"].abs()
    have["seqfar"] = have["seq_distance"] >= args.seq_far
    have["contact3d"] = have["dist_3d"] <= args.contact

    print("\n=== De-risking check: does 3D proximity add epistasis beyond sequence distance? ===")
    print("mean |epistasis| (n):")
    tab = have.groupby(["seqfar", "contact3d"]).agg(mean_abs_epi=("abs_epi", "mean"),
                                                    n=("abs_epi", "size"))
    for (sf, c3), row in tab.iterrows():
        sfl = "seq-FAR " if sf else "seq-near"
        c3l = "3D-contact " if c3 else "3D-distant"
        print(f"  {sfl} x {c3l}: {row['mean_abs_epi']:.3f}  (n={int(row['n']):,})")

    far = have[have["seqfar"]]
    if far["contact3d"].nunique() == 2:
        near_e = far.loc[far["contact3d"], "abs_epi"].mean()
        dist_e = far.loc[~far["contact3d"], "abs_epi"].mean()
        from scipy.stats import spearmanr, mannwhitneyu
        rho = spearmanr(far["dist_3d"], far["abs_epi"]).correlation
        try:
            u, pval = mannwhitneyu(far.loc[far["contact3d"], "abs_epi"],
                                   far.loc[~far["contact3d"], "abs_epi"], alternative="greater")
        except ValueError:
            pval = float("nan")
        print(f"\nAmong sequence-FAR pairs (|dpos| >= {args.seq_far}):")
        print(f"  mean |epi|  3D-contact {near_e:.3f}  vs  3D-distant {dist_e:.3f}")
        print(f"  Spearman(dist_3d, |epi|) = {rho:.3f}  (negative = closer means more epistasis)")
        print(f"  Mann-Whitney p (contact > distant) = {pval:.2e}")
        verdict = ("SUPPORTS a structure channel" if (near_e > dist_e and rho < 0)
                   else "does NOT clearly support a 3D channel -- reconsider M2 scope")
        print(f"  => {verdict}")

    print(f"\nWrote {args.out_pairs}" + (f" and {args.out_contacts}" if contacts else ""))


if __name__ == "__main__":
    main()
