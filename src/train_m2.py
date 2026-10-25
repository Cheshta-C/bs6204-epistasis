"""M2: structure-aware model for variant-pair epistasis.

M2 = M1 + structure. The ONLY additions over M1 are:
  * each residue's context vector comes from a GRAPH neural network that message-passes
    over the protein's 3D CONTACT graph (contacts.npz) plus sequence-adjacent edges,
    instead of the raw ESM embedding at that one position; so a mutated site now "sees"
    residues that are close in 3D even if far in sequence.
  * the 3D distance between the two mutated residues (dist_3d) is a head feature.
That isolates the contribution of structure: same frozen ESM node features, same
permutation-invariant head, same held-out-PROTEIN folds as M1.

The GNN is a plain dense GCN written in pure PyTorch -- NO torch-geometric, which is
painful to install on Colab. Each protein is small, so a dense LxL adjacency is cheap.

Predicts the epistasis residual; final prediction z_ab_hat = z_a + z_b + epistasis_hat.
Evaluated only on pairs that have a 3D distance (a structure); it also writes
pairs_struct.csv so M0/M1/M2 can be compared on that exact same subset.

Usage (new Colab notebook, same Drive folder; reuse the ESM cache from M1):
  pip install torch pandas numpy scipy
  python3 train_m2.py pairs_3d.csv sequences.csv contacts.npz \
      --cache wt_emb_650.pt --out m2_preds.csv
  # fair comparison on the structure-covered subset:
  python3 eval_harness.py pairs_struct.csv --pred m2_preds.csv --name M2_GNN
  python3 eval_harness.py pairs_struct.csv --pred m1_preds.csv --name M1_same_subset
  python3 eval_harness.py pairs_struct.csv                      # M0 on same subset

No-GPU smoke test (deterministic fake node features):
  python3 train_m2.py pairs_3d.csv sequences.csv contacts.npz --mock-embed --epochs 3
"""
import argparse
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

AA = "ACDEFGHIKLMNPQRSTVWYX"
AA_IDX = {a: i for i, a in enumerate(AA)}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mock_emb(seqs, dim=64):
    out = {}
    for _, r in seqs.iterrows():
        L = len(r["target_seq"])
        g = np.random.default_rng(int(hashlib.md5(r["assay"].encode()).hexdigest(), 16) % 2**32)
        out[r["assay"]] = torch.tensor(g.standard_normal((L, dim)), dtype=torch.float32)
    return out


def esm_cache(path):
    emb = torch.load(path)
    return {k: (v if torch.is_tensor(v) else torch.tensor(v)).float() for k, v in emb.items()}


def norm_adj(L, edges_1b):
    """Dense symmetric-normalised adjacency: contacts + sequence neighbours + self-loops."""
    A = np.eye(L, dtype=np.float32)
    idx = np.arange(L - 1)
    A[idx, idx + 1] = 1.0
    A[idx + 1, idx] = 1.0
    for a, b in edges_1b:
        ia, ib = int(a) - 1, int(b) - 1
        if 0 <= ia < L and 0 <= ib < L:
            A[ia, ib] = A[ib, ia] = 1.0
    d = A.sum(1)
    dinv = 1.0 / np.sqrt(np.clip(d, 1e-6, None))
    return torch.tensor(A * dinv[:, None] * dinv[None, :], dtype=torch.float32)


class M2(nn.Module):
    def __init__(self, emb_dim, hidden=128, aa_dim=16, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(emb_dim, hidden)
        self.gcn1 = nn.Linear(hidden, hidden)
        self.gcn2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.aa = nn.Embedding(len(AA), aa_dim)
        self.site = nn.Sequential(
            nn.Linear(hidden + 2 * aa_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 4, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def gnn(self, adj, x):
        h = torch.relu(self.in_proj(x))
        h = self.drop(torch.relu(adj @ self.gcn1(h)))
        h = torch.relu(adj @ self.gcn2(h))
        return h  # [L, hidden]  context-aware residue embeddings

    def pair(self, ctxA, wtA, mutA, ctxB, wtB, mutB, scalars):
        sA = self.site(torch.cat([ctxA, self.aa(wtA), self.aa(mutA)], -1))
        sB = self.site(torch.cat([ctxB, self.aa(wtB), self.aa(mutB)], -1))
        return self.head(torch.cat([sA + sB, scalars], -1)).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pairs_csv"); p.add_argument("sequences_csv"); p.add_argument("contacts_npz")
    p.add_argument("--cache", help="ESM per-residue embedding cache (.pt) from M1")
    p.add_argument("--mock-embed", action="store_true")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--out", default="m2_preds.csv")
    p.add_argument("--out-pairs", default="pairs_struct.csv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    pairs = pd.read_csv(args.pairs_csv)
    seqs = pd.read_csv(args.sequences_csv)
    contacts = {k: v for k, v in np.load(args.contacts_npz).items()}
    emb = mock_emb(seqs) if args.mock_embed else esm_cache(args.cache)

    # keep only pairs with a 3D distance AND a graph AND node features
    ok_assays = set(emb) & set(contacts)
    pairs = pairs[pairs["assay"].isin(ok_assays) & pairs["dist_3d"].notna()].reset_index(drop=True)
    print(f"M2 covers {len(pairs):,} pairs across {pairs['assay'].nunique()} proteins "
          f"(those with structure + embeddings).")
    pairs.to_csv(args.out_pairs, index=False)

    # precompute per-protein adjacency + node features on device
    adj, X = {}, {}
    for a in pairs["assay"].unique():
        Xa = emb[a].to(DEVICE)
        L = Xa.shape[0]
        adj[a] = norm_adj(L, contacts.get(a, np.empty((0, 2)))).to(DEVICE)
        X[a] = Xa
    emb_dim = next(iter(X.values())).shape[1]

    # tensors for the pair table
    def col_long(c): return torch.tensor(pairs[c].to_numpy(), dtype=torch.long)
    rowA = (pairs["pos_a"].to_numpy() - 1); rowB = (pairs["pos_b"].to_numpy() - 1)
    wtA = torch.tensor([AA_IDX.get(c, AA_IDX["X"]) for c in pairs["wt_a"]], dtype=torch.long)
    mutA = torch.tensor([AA_IDX.get(c, AA_IDX["X"]) for c in pairs["aa_a"]], dtype=torch.long)
    wtB = torch.tensor([AA_IDX.get(c, AA_IDX["X"]) for c in pairs["wt_b"]], dtype=torch.long)
    mutB = torch.tensor([AA_IDX.get(c, AA_IDX["X"]) for c in pairs["aa_b"]], dtype=torch.long)
    za = torch.tensor(pairs["z_a"].to_numpy(), dtype=torch.float32)
    zb = torch.tensor(pairs["z_b"].to_numpy(), dtype=torch.float32)
    d3 = torch.tensor((pairs["dist_3d"].to_numpy() / 10.0), dtype=torch.float32)
    sd = torch.tensor((pairs["seq_distance"].to_numpy() / 100.0).clip(0, 5), dtype=torch.float32)
    target = torch.tensor((pairs["z_ab"] - pairs["z_a"] - pairs["z_b"]).to_numpy(), dtype=torch.float32)
    assays = pairs["assay"].to_numpy()

    def forward_batch(model, b):
        """Compute predictions for the row indices in b, grouping by protein."""
        out = torch.empty(len(b), device=DEVICE)
        ba = assays[b]
        for a in np.unique(ba):
            m = np.where(ba == a)[0]          # positions within this batch
            gi = b[m]                          # global row indices
            H = model.gnn(adj[a], X[a])        # [L, hidden] for this protein
            ra = torch.tensor(rowA[gi], device=DEVICE)
            rb = torch.tensor(rowB[gi], device=DEVICE)
            scal = torch.stack([za[gi], zb[gi], d3[gi], sd[gi]], -1).to(DEVICE)
            pred = model.pair(H[ra], wtA[gi].to(DEVICE), mutA[gi].to(DEVICE),
                              H[rb], wtB[gi].to(DEVICE), mutB[gi].to(DEVICE), scal)
            out[torch.tensor(m, device=DEVICE)] = pred
        return out

    folds = sorted(pairs["fold"].unique())
    pred_resid = np.full(len(pairs), np.nan)
    for f in folds:
        te = np.where(pairs["fold"].to_numpy() == f)[0]
        tr = np.where(pairs["fold"].to_numpy() != f)[0]
        model = M2(emb_dim, hidden=args.hidden, dropout=args.dropout).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        bs = 2048
        for ep in range(args.epochs):
            model.train()
            order = np.random.permutation(tr)
            tot = 0.0
            for i in range(0, len(order), bs):
                b = order[i:i + bs]
                pred = forward_batch(model, b)
                loss = nn.functional.mse_loss(pred, target[b].to(DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            for i in range(0, len(te), bs):
                b = te[i:i + bs]
                pred_resid[b] = forward_batch(model, b).cpu().numpy()
        print(f"fold {f}: train {len(tr):,}  test {len(te):,}  (final train MSE {tot/len(tr):.3f})")

    out = pairs[["assay", "mutant"]].copy()
    out["pred_z_ab"] = pairs["z_a"].to_numpy() + pairs["z_b"].to_numpy() + pred_resid
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} and {args.out_pairs}. Compare on the SAME subset:")
    print(f"  python3 eval_harness.py {args.out_pairs} --pred {args.out} --name M2_GNN")
    print(f"  python3 eval_harness.py {args.out_pairs} --pred m1_preds.csv --name M1_same_subset")
    print(f"  python3 eval_harness.py {args.out_pairs}   # M0 floor on same subset")


if __name__ == "__main__":
    main()
