# Collocating or Interacting? Deep Learning of Variant-Pair Epistasis

BS6204 group project, NTU MSc Biomedical Data Science, 2026.
Team: CHHABRA CHESHTA, OMAR MOHAMED SULAIMAN,VIJAYAKUMAR SHAKTHI MAGESWARI, NURUL JANNA BINTE MUHAMMAD MUHIDDIN

We predict **epistasis**, meaning how much two mutations in the same protein interact beyond the
sum of their separate effects, for double mutants from deep mutational scanning (ProteinGym).
We compare three models:

| Model | What it is |
| --- | --- |
| M0 | Additive baseline: predicts zero interaction for every pair |
| M1 | Frozen ESM-2 (650M) per-residue embeddings + a small site encoder (summed over the two sites) + MLP head |
| M2 | M1 + a 2-layer graph convolutional network over AlphaFold2 residue contacts, plus 3D distance |

All models are evaluated on **held-out proteins** (5-fold cross-validation grouped by protein).
We also train a sequence-only ("no-singles") version of M1 that does not need measured
single-mutant effects, for scoring variants that have never been measured.

## Main results

| Result | Value |
| --- | --- |
| Data | 235,355 double mutants, 48 assays, 46 proteins |
| Epistasis Spearman, held-out proteins | M0 0 → M1 0.379 |
| RMSE on double-mutant z-score | M0 1.07 → M1 0.52 |
| Mean abs. epistasis, sequence-far pairs, 3D contact vs not | 0.407 vs 0.312 |
| M2 − M1 epistasis Spearman (3 seeds, 500 protein bootstraps) | −0.005, 95% CI −0.032 to 0.022 |
| No-singles M1, epistasis Spearman | 0.211 |

## Repository layout

```
src/
  build_pairs.py          build the pairs table, epistasis label and protein-grouped folds
  structure_features.py   AlphaFold2 contacts and 3D distances; model-free contact analysis
  train_m1.py             M1 (sequence model); also the no-singles variant
  train_m2.py             M2 (structure model, dense GCN in plain PyTorch)
  eval_harness.py         shared metrics for M0, M1, M2
  slice_eval.py           M1 vs M2 by pair geometry
  compare_ci.py           seed-averaged protein-bootstrap confidence intervals
results/                  output tables
notebooks/                Colab notebooks used to run training
```

## Setup

Python 3.10+. A GPU is needed to compute ESM-2 embeddings (we used Google Colab); the
training heads are small.

```bash
pip install -r requirements.txt
```

## Data (not included; download from ProteinGym)

From the ProteinGym website (https://proteingym.org, "Download"):

- `DMS_ProteinGym_substitutions.zip`: unzip to `data/DMS_ProteinGym_substitutions/`
- `DMS_substitutions.csv`: the assay reference file, saved to `data/`
- the AlphaFold2 structures archive (`ProteinGym_AF2_structures.zip`; the name may differ by release): unzip so the PDB files sit in `data/structures/`

## Reproducing the results

```bash
# 1. Pairs table, epistasis label, folds  ->  processed/
python src/build_pairs.py data/DMS_ProteinGym_substitutions data/DMS_substitutions.csv --out processed

# 2. M0 baseline
python src/eval_harness.py processed/pairs.csv

# 3. M1, three seeds (first run computes and caches ESM-2 embeddings)
for s in 0 1 2; do
  python src/train_m1.py processed/pairs.csv processed/sequences.csv \
      --esm-model esm2_t33_650M_UR50D --cache wt_emb_650.pt --seed $s --out m1_s$s.csv
done
python src/eval_harness.py processed/pairs.csv --pred m1_s0.csv --name M1

# 4. Structure features + model-free contact analysis
python src/structure_features.py processed/pairs.csv processed/sequences.csv data/structures \
    --out-pairs processed/pairs_3d.csv --out-contacts processed/contacts.npz

# 5. M2, three seeds (writes pairs_struct.csv, the structure-covered subset)
for s in 0 1 2; do
  python src/train_m2.py processed/pairs_3d.csv processed/sequences.csv processed/contacts.npz \
      --cache wt_emb_650.pt --seed $s --out m2_s$s.csv
done

# 6. Fair comparison on the same subset, slices, confidence intervals
python src/slice_eval.py pairs_struct.csv m1_s0.csv m2_s0.csv
python src/compare_ci.py pairs_struct.csv --m1 m1_s0.csv m1_s1.csv m1_s2.csv \
    --m2 m2_s0.csv m2_s1.csv m2_s2.csv --boot 500 --out ci_m2_vs_m1.tsv

# 7. No-singles model (CV score, then one final model trained on all pairs)
python src/train_m1.py processed/pairs.csv processed/sequences.csv --esm-model esm2_t33_650M_UR50D \
    --cache wt_emb_650.pt --no-singles --out m1_nosingles.csv
python src/train_m1.py processed/pairs.csv processed/sequences.csv --esm-model esm2_t33_650M_UR50D \
    --cache wt_emb_650.pt --no-singles --save-final m1_nosingles_final.pt
```

Every script also has a `--mock-embed` (or equivalent) smoke-test mode and a usage note at the top.

## Known issues and limitations

- **Label fitted on all doubles.** The per-assay z-scoring and the robust line used to define the
  epistasis label are fitted on all of an assay's doubles, including those of held-out proteins.
- **Not strictly order-invariant.** Site encodings are summed, but z(A) and z(B) enter the head in a
  fixed order.
- **Light model selection on the CV folds.** ESM-2 size and regularisation were compared on the same
  cross-validated scores; the M1 vs M2 comparison uses identical settings.
- Training proteins are small (median about 58 residues), so pairs far apart in long proteins are
  outside the training range.

## References

- Notin P et al. ProteinGym. NeurIPS Datasets and Benchmarks (2023).
- Lin Z et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science (2023).
- Jumper J et al. Highly accurate protein structure prediction with AlphaFold. Nature (2021).
- Rao R et al. Transformer protein language models are unsupervised structure learners. ICLR (2021).
- Kipf TN, Welling M. Semi-supervised classification with graph convolutional networks. ICLR (2017).
