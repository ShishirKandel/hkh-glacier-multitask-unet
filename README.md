# Multi-Task U-Net for Himalayan Glacier Mapping

From-scratch PyTorch implementation of a three-headed U-Net that maps glacier
extent, discriminates clean from debris-covered ice, and regresses distance to
the nearest glacier boundary, trained on the
[HKH Glacier Mapping dataset](https://lila.science/datasets/hkh-glacier-mapping/)
(Landsat-7 + SRTM, expert labels by ICIMOD).

Coursework project for STW7088CEM Artificial Neural Networks (Softwarica /
Coventry University). Author: Shishir Kandel (250619).

![Multi-task U-Net architecture](docs/fig_architecture.png)

## Results

Trained under a frozen, zero-leakage geographic split (verified zero footprint
overlap, 0.15 deg train buffer). Test and east zones were each evaluated exactly
once, after all model selection.

| Zone | Glacier IoU | Debris IoU | Macro-F1 | Dist MAE |
|---|---|---|---|---|
| Validation (best rung f) | 0.665 | 0.259 | 0.732 | 4.4 px |
| Validation (final checkpoint) | 0.655 | 0.234 | 0.716 | 4.3 px |
| Test, unseen region (one-shot) | 0.402 | 0.132 | 0.595 | 5.2 px |
| East, OOD (one-shot) | 0.494 | 0.193 | 0.665 | 2.6 px |

The performance is attributed, not just obtained: an eleven-rung ablation ladder
(configs `a` through `final`) adds one change at a time. Key findings: terrain
channels produce the first debris signal; Dice loss plus debris oversampling
amplify it; the auxiliary boundary-distance regression head improves both
classification tasks; a focal/pos-weight stack on top of Dice actively hurts
(isolated in rung `d2b`).

## Repository layout

```
data.py            shard-backed dataset: dequantisation, train-stats normalisation,
                   validity masking, crops/flips/rot90 + spectral jitter
model.py           U-Net from scratch; optional attention gates; prior-logit head init
losses.py          masked BCE/Dice/focal + masked L1 distance loss
train.py           training loop, smoothed checkpoint selection, metrics.jsonl logging
eval.py            one-shot masked evaluation (confusion matrix, IoU/Dice/F1, dist MAE)
configs/           one YAML per ablation rung (a, b1, b2, c, d1, d2, d2b, e1, e2,
                   f, final) plus rehearse_a
figures/           figure generation scripts (sample patch, split map, architecture
                   diagram, ablation ladder, training curves, qualitative montage)
analysis/          the frozen split contract, the run record, and the evaluation
                   outputs every reported number is taken from (see below)
```

The bulk data is not in this repository and cannot be: the source archive is 74 GB
compressed, and the curated shards, extracted patches and checkpoints come to
roughly 90 GB. What is here is everything needed to check the claims.

```
analysis/split.json               freeze contract: bands, assignment rules, hashed
                                  evaluation lists, full per-zone membership
analysis/subset_manifest.csv      all 3,168 candidate patches with zone, scene,
                                  lon/lat, class pixel counts and validity
analysis/shards/split_meta.json   independent copy of the frozen lists, as uploaded
                                  to Kaggle, for cross-checking split.json
analysis/shards/quant_spec.json   per-band uint16 quantisation spec
analysis/shards/train_stats.json  train-zone normalisation statistics
analysis/archive_sha256.txt       SHA-256 of the source archive
analysis/geo/                     patch footprint centroids and slice geometry
analysis/kaggle_notebook/         the notebook that executed every rung, and its
                                  kernel metadata (T4, no internet, seeded)
analysis/runs_kaggle/<rung>/      per-rung outputs, one directory per training run
```

## Verifying the reported numbers

Every figure in the results table above can be recomputed from this repository
without a GPU. Each `analysis/runs_kaggle/<rung>/` holds:

- `eval_val.json` -- the validation evaluation the rung's results row is taken
  from, with the full 3x3 confusion matrix, per-class IoU/Dice/precision/recall,
  macro-F1 and distance MAE. `final/` additionally holds `eval_test.json` and
  `eval_east.json`, the two one-shot held-out evaluations.
- `metrics.jsonl` -- one record per epoch, the series the training-curve and
  ablation figures are drawn from.
- `kernel.log` -- the Kaggle kernel's own stdout for that run.

Two directories are **superseded runs, kept deliberately** rather than deleted:
`a_v2` is the all-background collapse (validation glacier IoU 0.011) that motivated
prior-logit head initialisation, and `b1` is the pre-fix run whose checkpoint was
chosen by an epoch-1 spike, which motivated smoothed selection with a minimum-epoch
guard. Both failures are discussed in the report. The rungs the results table
reports are `a_v3` and `b1_v5`.

## Reproducing

1. Download `hkh_patches.tar.gz` from the LILA link above (74 GB).
2. Curate and shard: filter patches by validity, uint16-quantise per band, pack
   per-scene npz shards with distance-transform targets (the pipeline, split
   contract and SHA-256 checksums are documented in the report's reproduction
   appendix).
3. Train a rung: `python train.py --config configs/f.yaml --data <shards>`
4. Evaluate: `python eval.py --ckpt runs/f/f/best.pt --zone test --data <shards>`

All runs used seed 20260716 on a Kaggle T4 (the whole ladder costs about 3.5
GPU-hours). The exact per-rung logs and device-visible run evidence are in the
report appendices.
