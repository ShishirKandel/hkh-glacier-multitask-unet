# HKH Glacier Mapping — Curated Training Shards

Curated, quantized subset of the HKH Glacier Mapping dataset (LILA BC, CC BY 4.0) for the
STW7088CEM multi-task U-Net project. Built from `hkh_patches.tar.gz`
(SHA-256 `627dff4b0a9635cdb5d7868495552a707bb718fa33905d1bd7a4d96c3705e935`) via a frozen,
leakage-free geographic split (see `split_meta.json` — bands, clauses, eval-list hashes).

## Layout
```
{train,val,test,east}/shard_sNN.npz   one shard per source Landsat scene NN per zone
train_stats.json                      per-band mean/std (TRAIN only, valid pixels)
quant_spec.json                       uint16 dequantization spec
split_meta.json                       frozen split contract summary
```

## Shard contents
| key | shape | dtype | meaning |
|---|---|---|---|
| `imgs` | (N,512,512,15) | uint16 | quantized channels; **65535 = NaN/nodata** |
| `masks` | (N,512,512,2) | uint8 | canonical labels: ch0=clean ice, ch1=debris-covered |
| `dist` | (N,512,512) | int8 | truncated signed distance to glacier boundary, +inside/−outside, clipped [−32,31] |
| `names` | (N,) | str | original patch basenames |

Channels 0–14: B1,B2,B3,B4,B5,B6a,B6b,B7,B8(pan),BQA,NDVI,−NDSI,NDWI,elevation(m),slope(°).
Note ch11 is sign-flipped NDSI (snow/ice negative).

## Loader
```python
import numpy as np, json
spec = json.load(open("quant_spec.json"))
qmin, scale = np.array(spec["qmin"]), np.array(spec["scale"])
z = np.load("train/shard_s08.npz")
q = z["imgs"][0].astype(np.float32)
img = qmin + q * scale          # dequantize
img[z["imgs"][0] == 65535] = np.nan  # nodata
glacier = (z["masks"][0].sum(-1) > 0)  # union head target
```

## Binding rules (from the frozen split contract)
1. Mask nodata **per pixel** (finite in all 15 channels) in the loss and every metric.
2. Normalize with `train_stats.json` (train-only statistics).
3. `test/` is evaluated exactly once; `east/` is OOD-reporting only (never selection).
