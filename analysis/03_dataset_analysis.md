# HKH Glacier Mapping Dataset — Full Analysis Report

*Produced 2026-07-16 from a complete streaming scan of `hkh_patches.tar.gz` (73.86 GB compressed,
182.39 GB uncompressed, 15,384 members, 0 corrupt files, scanned in 15.5 min). All numbers below
are observed, not quoted from documentation.*

---

## 1. Archive inventory

| Directory | Contents | Count | Uncompressed size |
|---|---|---|---|
| `glacier_data/slices/` | Patch pairs `slice_S_img_N.npy` + `slice_S_mask_N.npy` | **7,095 pairs** (14,190 files) | ~117.2 GB |
| `glacier_data/slices/slices.geojson` | **Per-patch footprint polygons + source-scene metadata** | 7,095 features | 5.4 MB |
| `glacier_data/slices/stats_train.json` | Authors' per-band train-split mean/std (15 values each) | 1 | 606 B |
| `glacier_data/splits/{train,dev,test}/` | Official filtered splits (≥5% glacier recipe per README) | **383 / 55 / 110 pairs** | ~8.9 GB |
| `glacier_data/raster_data/` | Original georeferenced Landsat-7 scenes `LE07_PPPRRR_YYYYMMDD.tif` | 35 | ~50 GB |
| `glacier_data/masks/` | Full-tile masks `mask_00..34.npy` (+1 aux file) | 36 | ~5.9 GB |
| `glacier_data/vector_data/` | ICIMOD glacier polygons: `clean.shp` (28,570), `debris.shp` (1,526), `hkh.shp` (6 Natural-Earth country boundaries) | 15 files | ~189 MB |
| `glacier_data/README.md` | Dataset documentation (partially inaccurate — see §3) | 1 | 1.6 KB |

**The proposal's "14,190 patches" counts img+mask files together; the true unit is 7,095 pairs.**
The 548 split pairs are duplicated copies (by basename) of `slices/` patches.

## 2. Patch format and channel map (all verified empirically)

**Images:** `(512, 512, 15)` float32. Channel identification (indices recomputed from bands with
correlation 1.0 and max|diff| 0.0; thermal pair correlate at 0.9995; pan vs visible 0.98):

| ch | Content | Observed range notes |
|---|---|---|
| 0–4 | Landsat-7 B1 (blue), B2 (green), B3 (red), B4 (NIR), B5 (SWIR1) | 8-bit DN scale, but resampling overshoot produces values outside [0,255] (min ≈ −262, max ≈ 441) |
| 5–6 | B6_VCID_1, B6_VCID_2 (thermal, two gains) | compressed ranges; corr 0.9995 |
| 7 | B7 (SWIR2) | as 0–4 |
| 8 | B8 (panchromatic) | as 0–4 |
| 9 | BQA quality bitmask | **interpolated to non-integer values** — must round before decoding bits |
| 10 | NDVI = (B4−B3)/(B4+B3) | exact; outliers beyond ±1 from negative DNs (min −7.8, max 9.4) |
| 11 | **−NDSI** = (B5−B2)/(B5+B2) | exact but **sign-flipped**: snow/ice is NEGATIVE here |
| 12 | NDWI(Gao)/NDMI = (B4−B5)/(B4+B5) | exact |
| 13 | SRTM elevation (m) | mean 2,759 m, max 7,458 m; small negative artifacts |
| 14 | Slope (degrees) | 0–81°, mean 22° |

**Masks:** `(512, 512, 3)` uint8 {0,1} — **two conventions coexist**:

- `slices/` (7,095): **ch0 = all glacier (union), ch1 = clean ice, ch2 = debris-covered ice**.
  Union identity `ch0 = ch1 + ch2` holds exactly in 7,080/7,095 masks. The 15 exceptions have
  1–1,509 px labelled both clean and debris (double-labelled overlap — negligible label noise,
  worth one report sentence).
- `splits/` (548): **one-hot (clean, debris, background)** — the three channels sum to exactly 1
  per pixel. Their ch0/ch1 equal the same-named slices mask's ch1/ch2 verbatim.

The README's channel description (clean, debris, HKH-validity) matches **neither** convention —
document this in the report's Data section. **Debris supervision exists and RQ2 is trainable.**

## 3. README inaccuracies (for the report's Data section)

1. Mask channels are not (clean, debris, HKH-validity) — see §2.
2. `hkh.shp` is not the ICIMOD Glacier_2005 inventory — it is a 6-country Natural Earth
   boundary layer (encoding-corrupt DBF), usable for figures only.
3. The glacier inventory attributes live in `clean.shp`/`debris.shp`: GLIMS_ID, Basin, M_Basin,
   Sub_basin, Region, Country, Elv_min/mean/max, slope stats, Aspect, Area_SqKm, Thickness.
   Bounding box 67.6–98.2°E, 27.5–38.3°N — the whole HKH arc, **not just Nepal and Bhutan**.

## 4. Class statistics (slices/ only — clean numbers)

Corpus: 7,095 patches = 1,859,911,680 pixels.

| Class | Pixels | % of all pixels | Patches containing it | % of patches |
|---|---|---|---|---|
| Glacier (union) | 45,014,441 | 2.420% | 1,970 | 27.8% |
| — clean ice | 40,549,176 | 2.180% | 1,969 | 27.8% |
| — debris-covered | 4,469,018 | **0.240%** | 928 | 13.1% |
| Background only | — | 97.58% | 5,125 | 72.2% |

Imbalance ratios: glacier:background ≈ 1:40 · debris:all ≈ **1:416** · debris:clean ≈ 1:9.1.
This confirms (and quantifies) the proposal's "small fraction / smaller fraction still" claim.

**Per-slice distribution** (full table in `slice_table.csv`): glacier fraction ranges
0.008%–9.99% per slice, debris 0%–1.47%. **Four slices have zero debris** (18, 19, 30, 33) and
five more are near-zero; debris concentrates in slices 16, 12, 23, 15, 8, 9. Split design must
allocate debris-rich slices to every partition or the test set becomes RQ2-unevaluable.

## 5. Geolocation and splits

- **`slices.geojson` gives every patch a projected footprint polygon** plus its source scene and
  tile mask. The proposal's "geographic split" is therefore fully feasible (the panel's
  scene-level fallback is no longer needed — true spatial clustering is possible).
- Slice → scene mapping is 1:1 for all 35 slices (e.g., slice_0 = LE07_149037_20041024).
  Path/row spans 134–153 / 33–41 across the HKH arc; adjacent paths sidelap, so spatially-aware
  assignment (from the geojson footprints) is still advised.
- Scene dates: 2001×2, 2004×2, 2005×6, 2006×13, 2007×7, 2008×2, 2009×1 —
  **33/35 scenes are post-May-2003, i.e., SLC-off** (striping gaps), consistent with the ~49%
  NaN fraction in spectral channels (sampled estimate over 342 images; includes whole-patch
  nodata at scene edges — e.g., slice_0_img_014 is 100% NaN in ch0–12). Terrain channels are
  ~36% NaN. **Per-patch valid-pixel fraction must be computed during the reduction pass and
  wholly-invalid patches dropped.**
- **Official splits included**: 383 train / 55 dev / 110 test pairs (lists exported to
  `official_splits.json`). These implement the README's ≥5%-glacier filter. They are small —
  the project can (a) train on them for comparability with the dataset authors, and/or
  (b) build its own larger split from the 1,970 glacier patches + background sample, justified
  in the report. `stats_train.json` gives the authors' normalization stats (its ch11 mean of
  −0.535 independently confirms the sign-flipped NDSI).

## 6. Data-quality findings (each one is report material)

1. ~49% of spectral pixels NaN (SLC-off striping + scene-edge nodata) — sampled estimate.
2. Some patches 100% NaN in all spectral channels — must be filtered.
3. DN bands contain out-of-range values (negative, >255) from resampling — clip or robust-normalize.
4. Index channels have extreme outliers (NDVI −7.8..9.4) — clip to [−1,1] or recompute from bands.
5. BQA is interpolated (non-integer) — round before decoding; snow/ice-confidence bits could even
   serve as an auxiliary input or sanity check.
6. ch11 is sign-flipped NDSI — handle in code and document.
7. 15 masks with 1–1.5k px clean/debris double-labels — negligible, disclose.
8. 548 `analysis/masks/` files on local disk hold the one-hot splits version (same info,
   different layout) — any local loader must detect convention (one-hot rows sum to 1).

## 7. Corrections the report must make to proposal claims

1. "14,190 patches" → 7,095 img/mask pairs (14,190 files).
2. Masks 512×512×2 → 512×512×3, channels (union, clean, debris); README also wrong.
3. Coverage "Nepal and Bhutan" → whole HKH arc (67.6–98.2°E) per shipped shapefiles.
4. "Colab/Kaggle GPU" on 182 GB → requires a local curate/quantize/shard reduction stage first.
5. Geographic split promise → fully achievable via slices.geojson (better than expected).
6. RQ1 "from Landsat-7 imagery" → inputs also include SRTM terrain channels (clarify).
7. Dataset link/license: cite from the LILA page (the one permitted external link).

## 8. Environment constraints (unchanged)

Local: Windows 11, Python 3.13, torch CPU-only, E: 63.9 GB free (after extracting ~56 GB of
rasters/tile-masks). Training must happen on Kaggle (primary) / Colab (fallback) per the
execution plan in `02_execution_plan.md`. The ~50 GB `analysis/extracted/raster_data/` can be
deleted to reclaim space once its role (georeferencing/figures) is decided — the tar.gz still
holds the originals.

## 9. Artifacts produced by this analysis

| File | Purpose |
|---|---|
| `manifest.jsonl` | Every archive member: name, size, per-mask class counts, shapes |
| `summary.json` | Raw scan aggregates (NOTE: its mask_totals/slices sections mix splits+slices; superseded by §4 and `slice_table.csv`) |
| `slice_table.csv` | Clean per-slice: scene ID, pairs, glacier%, debris%, patch counts |
| `official_splits.json` | Official train/dev/test patch lists |
| `masks/` (7,095 npy) | All masks locally — split design & distance-transform targets without re-reading the tar |
| `samples/` (~95 pairs) | Random img+mask pairs for visualization/prototyping |
| `extracted/` | README, vector_data, slices.geojson, stats_train.json, raster_data (35 GeoTIFF), tile masks |
| `fig_sample_patch.png` | 8-panel montage of a debris-rich patch (bands/terrain/labels) |
| `00_project_memo.md` / `01_compliance_audit.md` / `02_execution_plan.md` | Reconciliation memo, compliance audit, marking-criteria-mapped execution plan |

## 10. Answers to the memo's open questions (§5 of `00_project_memo.md`)

| Question | Answer |
|---|---|
| Mask semantics | RESOLVED: (union, clean, debris) in slices/; one-hot in splits/; debris supervision exists → **training is unblocked** |
| Exact patch count | 7,095 pairs (proposal's 14,190 = files, not pairs) |
| Per-slice aggregates | `slice_table.csv`; 4 debris-free slices; debris concentrated in ~6 slices |
| Band statistics | §2 table; stats_train.json provides authors' normalization |
| slices.geojson present? | YES → true geographic split feasible; also raster_data + tile masks present |
| Archive checksum | Still to compute (SHA-256 of the 74 GB tar.gz — ~5 min job, do before report submission) |
