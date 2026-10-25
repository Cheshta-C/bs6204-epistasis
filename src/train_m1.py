"""M1: ESM-2 sequence model for variant-pair epistasis.

Design (why it is built this way):
  * The model predicts the EPISTASIS RESIDUAL, not the raw double-mutant effect.
    Final prediction is  z_ab_hat = z_a + z_b + epistasis_hat.
    So all model capacity goes to the interaction M0 cannot capture. (The model itself
    fits nothing per assay; note that the epistasis LABEL and the evaluation baseline in
    build_pairs.py / eval_harness.py are fitted per assay on all doubles, see README.)
  * ESM-2 is used FROZEN as a feature extractor. Each wild-type sequence is embedded
    ONCE (there are only ~48), per-residue vectors cached to disk. Training then only
    touches a small head, so all 5 folds run quickly.
  * The two SITE encodings are SUMMED, so the site part of the model does not depend on
    which mutation is called A or B. Note: z_a and z_b enter the head in a fixed order,
    so the full model is not strictly order-invariant (see README, Known issues).

Per-fold held-out-PROTEIN cross-validation: train on 4 folds, predict the 5th, repeat,
then concatenate. Output is a predictions CSV in the exact format eval_harness.py wants:
    assay, mutant, pred_z_ab

Usage on the GPU machine:
  pip install fair-esm torch pandas numpy
  python3 train_m1.py processed/pairs.csv processed/sequences.csv \
      --esm-model esm2_t12_35M_UR50D --out m1_preds.csv
  python3 eval_harness.py processed/pairs.csv --pred m1_preds.csv --name M1_ESM35

Smoke test without ESM or a GPU (deterministic fake embeddings, proves the plumbing):
  python3 train_m1.py processed/pairs.csv processed/sequences.csv --mock-embed --epochs 3
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


# ---------- embedding backend ----------
def esm_wt_embeddings(seqs, model_name, cache_path):
    """Return {assay: tensor[L, D]} of frozen per-residue ESM-2 embeddings. Cached."""
    import os
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached WT embeddings from {cache_path}")
        return torch.load(cache_path)
    import esm
    print(f"Loading {model_name} ...")
    model, alphabet = esm.pretrained.__dict__[model_name]()
    bc = alphabet.get_batch_converter()
    model = model.to(DEVICE).eval()
    layer = model.num_layers
    out = {}
    with torch.no_grad():
        for _, r in seqs.iterrows():
            data = [(r["assay"], r["target_seq"])]
            _, _, toks = bc(data)
            toks = toks.to(DEVICE)
            rep = model(toks, repr_layers=[layer])["representations"][layer]
            # drop BOS/EOS -> [L, D], back to cpu to save GPU memory
            out[r["assay"]] = rep[0, 1:len(r["target_seq"]) + 1].cpu()
            print(f"  embedded {r['assay']}  L={len(r['target_seq'])}  D={rep.shape[-1]}")
    if cache_path:
        torch.save(out, cache_path)
    return out


def mock_wt_embeddings(seqs, dim=64):
    """Deterministic fake embeddings for a no-GPU smoke test of the pipeline."""
    out = {}
    for _, r in seqs.iterrows():
        L = len(r["target_seq"])
        g = np.random.default_rng(int(hashlib.md5(r["assay"].encode()).hexdigest(), 16) % 2**32)
        out[r["assay"]] = torch.tensor(g.standard_normal((L, dim)), dtype=torch.float32)
    return out


# ---------- model ----------
class SiteEncoder(nn.Module):
    def __init__(self, emb_dim, aa_dim=16, hidden=128, dropout=0.1):
        super().__init__()
        self.aa = nn.Embedding(len(AA), aa_dim)
        self.net = nn.Sequential(
            nn.Linear(emb_dim + 2 * aa_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, ctx, wt, mut):
        x = torch.cat([ctx, self.aa(wt), self.aa(mut)], dim=-1)
        return self.net(x)


class EpistasisHead(nn.Module):
    def __init__(self, emb_dim, hidden=128, dropout=0.1):
        super().__init__()
        self.site = SiteEncoder(emb_dim, hidden=hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, ctxA, wtA, mutA, ctxB, wtB, mutB, za, zb, dist):
        # SUM the two site encodings -> permutation invariance in the pair
        s = self.site(ctxA, wtA, mutA) + self.site(ctxB, wtB, mutB)
        pair = torch.cat([s, za[:, None], zb[:, None], dist[:, None]], dim=-1)
        return self.head(pair).squeeze(-1)


# ---------- feature assembly ----------
def build_tensors(df, emb):
    ctxA, ctxB, wtA, mutA, wtB, mutB = [], [], [], [], [], []
    for _, r in df.iterrows():
        e = emb[r["assay"]]
        ctxA.append(e[int(r["pos_a"]) - 1])
        ctxB.append(e[int(r["pos_b"]) - 1])
        wtA.append(AA_IDX.get(r["wt_a"], AA_IDX["X"]))
        mutA.append(AA_IDX.get(r["aa_a"], AA_IDX["X"]))
        wtB.append(AA_IDX.get(r["wt_b"], AA_IDX["X"]))
        mutB.append(AA_IDX.get(r["aa_b"], AA_IDX["X"]))
    t = lambda x: torch.tensor(np.array(x), dtype=torch.float32)
    L = lambda x: torch.tensor(x, dtype=torch.long)
    return dict(
        ctxA=torch.stack(ctxA), ctxB=torch.stack(ctxB),
        wtA=L(wtA), mutA=L(mutA), wtB=L(wtB), mutB=L(mutB),
        za=t(df["z_a"].to_numpy()), zb=t(df["z_b"].to_numpy()),
        dist=t((df["seq_distance"].to_numpy() / 100.0).clip(0, 5)),
        target=t((df["z_ab"] - df["z_a"] - df["z_b"]).to_numpy()),  # naive epistasis residual
    )


def run_epoch(model, feats, idx, opt=None, bs=4096):
    train = opt is not None
    model.train(train)
    order = np.random.permutation(idx) if train else idx
    total, n = 0.0, 0
    for i in range(0, len(order), bs):
        b = order[i:i + bs]
        args = [feats[k][b].to(DEVICE) for k in
                ("ctxA", "wtA", "mutA", "ctxB", "wtB", "mutB", "za", "zb", "dist")]
        pred = model(*args)
        y = feats["target"][b].to(DEVICE)
        loss = nn.functional.mse_loss(pred, y)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(b); n += len(b)
    return total / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pairs_csv")
    p.add_argument("sequences_csv")
    p.add_argument("--esm-model", default="esm2_t12_35M_UR50D")
    p.add_argument("--cache", default="wt_emb.pt")
    p.add_argument("--mock-embed", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.1, help="raise (e.g. 0.3) if overfitting")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="L2; raise (e.g. 1e-3) if overfitting")
    p.add_argument("--out", default="m1_preds.csv")
    p.add_argument("--save-final", help="train ONE model on ALL pairs (no CV) and save it here, for scoring new proteins")
    p.add_argument("--no-singles", action="store_true",
                   help="hide the measured single-mutant effects z_a/z_b from the model, so it can score natural variants")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    pairs = pd.read_csv(args.pairs_csv)
    seqs = pd.read_csv(args.sequences_csv)

    emb = (mock_wt_embeddings(seqs) if args.mock_embed
           else esm_wt_embeddings(seqs, args.esm_model, args.cache))
    emb_dim = next(iter(emb.values())).shape[1]
    print(f"Embedding dim = {emb_dim}; building features for {len(pairs)} pairs ...")
    feats = build_tensors(pairs, emb)
    if args.no_singles:
        # The model never sees z_a/z_b. The raw residual z_ab - z_a - z_b carries a large
        # assay-level bias (saturating readouts) that only the singles can explain, so train
        # instead on the FITTED epistasis label (z_ab minus the per-assay robust additive fit),
        # which is centred on zero and needs no single effects at prediction time.
        feats["za"] = torch.zeros_like(feats["za"])
        feats["zb"] = torch.zeros_like(feats["zb"])
        feats["target"] = torch.tensor(pairs["epistasis"].to_numpy(), dtype=torch.float32)
        print("no-singles mode: z_a/z_b hidden; target = fitted epistasis label")

    if args.save_final:
        # train a single model on ALL pairs, for applying to new proteins
        model = EpistasisHead(emb_dim, dropout=args.dropout).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        allidx = np.arange(len(pairs))
        for ep in range(args.epochs):
            tl = run_epoch(model, feats, allidx, opt)
        torch.save({"state_dict": model.state_dict(), "emb_dim": emb_dim,
                    "hidden": 128, "aa": AA, "esm_model": args.esm_model,
                    "no_singles": args.no_singles, "dropout": args.dropout,
                    "target": "fitted_epistasis" if args.no_singles else "naive_residual"},
                   args.save_final)
        print(f"Saved final M1 (trained on all {len(pairs):,} pairs) to {args.save_final} "
              f"(final train MSE {tl:.3f})")
        return

    folds = sorted(pairs["fold"].unique())
    pred_resid = np.full(len(pairs), np.nan)
    for f in folds:
        te = np.where(pairs["fold"].to_numpy() == f)[0]
        tr = np.where(pairs["fold"].to_numpy() != f)[0]
        model = EpistasisHead(emb_dim, dropout=args.dropout).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for ep in range(args.epochs):
            tl = run_epoch(model, feats, tr, opt)
        model.eval()
        with torch.no_grad():
            args_te = [feats[k][te].to(DEVICE) for k in
                       ("ctxA", "wtA", "mutA", "ctxB", "wtB", "mutB", "za", "zb", "dist")]
            pred_resid[te] = model(*args_te).cpu().numpy()
        print(f"fold {f}: trained on {len(tr)}, predicted {len(te)}  (final train MSE {tl:.3f})")

    out = pairs[["assay", "mutant"]].copy()
    if args.no_singles:
        # prediction is on the fitted-epistasis scale; rebuild z_ab around the additive fit so
        # the harness's implied epistasis (pred_z_ab - z_additive_fit) equals the prediction itself
        out["pred_z_ab"] = pairs["z_additive_fit"].to_numpy() + pred_resid
    else:
        out["pred_z_ab"] = pairs["z_a"].to_numpy() + pairs["z_b"].to_numpy() + pred_resid
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}. Now score it:")
    print(f"  python3 eval_harness.py {args.pairs_csv} --pred {args.out} --name M1_ESM")


if __name__ == "__main__":
    main()
