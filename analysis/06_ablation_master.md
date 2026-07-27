# Master Ablation Table — validation zone (283 patches), best smoothed checkpoint

All rungs: seed 20260716, Adam lr 3e-4, batch 8, 256² crops, ≤30 epochs (patience 8),
prior-logit head init, per-pixel validity masking, train-only normalization.
Selection: mean(glacier IoU, debris IoU) smoothed over last 3 evals.
TEST zone untouched until final config; EAST reserved for final OOD report.

| Rung | Config delta | Glacier IoU | Glacier Dice | Glacier P/R | Debris IoU | Debris P/R | Macro-F1 | Best ep | GPU min |
|---|---|---|---|---|---|---|---|---|---|
| a | visible bands (B1-B3), plain U-Net, unweighted BCE | 0.322 | 0.487 | 0.90 / 0.33 | 0.000 | 0.00 / 0.00 | 0.489 | 19/27 | ~16 |
| b1 | + NIR/SWIR/thermal (ch 3-7) | 0.438 | 0.609 | 0.92 / 0.46 | 0.000 | 0.00 / 0.00 | 0.533 | 11/19 | ~12 |
| b2 | + elevation & slope (ch 13-14) | 0.568 | 0.724 | 0.84 / 0.63 | 0.066 | 0.39 / 0.07 | 0.611 | 15/23 | ~20 |
| c | + geometric/spectral augmentation | 0.627 | 0.771 | 0.85 / 0.70 | 0.013 | 0.20 / 0.01 | 0.596 | 30/30 | ~22 |
| d1 | + Dice added to BCE (both heads) | 0.602 | 0.752 | 0.70 / 0.81 | 0.171 | 0.26 / 0.34 | 0.672 | 15/23 | ~20 |
| d2 | + focal(type), pos_weight auto (7.69), debris oversample ×3, head wt 1.5 | 0.596 | 0.747 | 0.81 / 0.69 | 0.106 | 0.18 / 0.20 | 0.631 | 12/20 | ~18 |
| d2b | d1 + oversample ×3 + head wt 1.5 only (isolate: no focal/pos_weight) | 0.612 | 0.760 | 0.74 / 0.77 | **0.196** | 0.33 / 0.32 | **0.685** | 12/20 | ~18 |
| e1 | d2b + attention gates | 0.584 | 0.737 | 0.67 / 0.82 | 0.210 | 0.34 / 0.35 | 0.681 | 19/27 | ~24 |
| e2 | e1 + dropout 0.2 | 0.575 | 0.730 | 0.79 / 0.68 | 0.190 | 0.27 / 0.40 | 0.672 | 6/15 | ~14 |
| f | d2b + boundary-distance regression head | **0.665** | **0.799** | 0.81 / 0.78 | **0.259** | 0.39 / 0.44 | **0.732** | 30/30 | ~24 |
| final | f config, 60-ep schedule (42 trained, best ep 32) | 0.655 | 0.792 | 0.80 / 0.78 | 0.234 | 0.48 / 0.31 | 0.716 | 32/42 | ~31 |

## Final model — held-out evaluations (each run EXACTLY ONCE, per frozen contract)

| Zone | Glacier IoU | Glacier Dice | Glacier P/R | Debris IoU | Debris P/R | Macro-F1 | Dist MAE | n |
|---|---|---|---|---|---|---|---|---|
| test (west band, 72.95-74.65°E) | 0.402 | 0.573 | 0.61 / 0.54 | 0.132 | 0.17 / 0.38 | 0.595 | 5.23 px | 315 |
| east (≥86°E, OOD) | 0.494 | 0.661 | 0.68 / 0.65 | 0.193 | 0.63 / 0.22 | 0.665 | 2.62 px | 293 |

Notes: the val→test gap (0.655→0.402 glacier IoU) is the honest price of a zero-leakage
geographic split — test is the debris-rich Karakoram sector, a genuinely different
glaciological regime. The east OOD set transfers BETTER than the contiguous test band
(0.494 IoU, debris precision 0.63): eastern-Himalayan glaciers are more similar to the
central-Himalayan training areas than the Karakoram is, despite greater distance.
Both numbers were produced once, after all model selection, and cannot have been tuned.

## Rung (a) notes
- Kernel v3. Collapse-free after prior-init fix (see 05_experiment_log.md); val
  trajectory noisy (IoU 0.10-0.46 epoch-to-epoch) — BN train/eval statistics on
  256² crops vs full 512² eval + threshold hover; smoothed selection handles it.
- High precision / low recall = conservative under-segmentation, classic unweighted-BCE
  behaviour under imbalance. Debris entirely missed — the motivating failure for rung d.
- Confusion matrix (bg/clean/debris, pixels): debris row = [436580, 5393, 0].
