"""Shard-backed PyTorch dataset for the HKH glacier multi-task U-Net.

Shard format (analysis/shards/README.md): {train,val,test,east}/shard_sNN.npz with
imgs uint16 (N,512,512,15; 65535 = nodata), masks uint8 (N,512,512,2) = clean,debris,
dist int8 (N,512,512), names str.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# Canonical channel groups (indices into the 15-band stack
# B1,B2,B3,B4,B5,B6a,B6b,B7,B8(pan),BQA,NDVI,-NDSI,NDWI,elevation,slope):
#   visible          = [0, 1, 2]
#   nir_swir_thermal = [3, 4, 5, 6, 7]
#   pan_bqa          = [8, 9]       (pan duplicates VIS; BQA is a quality bitmask)
#   indices          = [10, 11, 12] (derivable from the reflectance bands)
#   terrain          = [13, 14]
NODATA = 65535


class ShardDataset(Dataset):
    """One zone of the curated shards.

    train=True enables random crop / flips / rot90 (plus per-band spectral jitter
    when config 'aug' is true); eval mode returns full 512x512 patches, no aug.
    config 'mmap': false (default) preloads the uint16 arrays into RAM at init so
    forked DataLoader workers share the pages (~8 GB for train); true re-opens the
    npz on every access to save RAM (slower; also the safe choice on Windows,
    where spawn-based workers would copy the preloaded cache).
    """

    def __init__(self, root, zone, config, train=False):
        self.root, self.zone, self.train = Path(root), zone, train
        self.channels = list(config.get("channels", range(15)))
        self.crop = int(config.get("crop", 256))
        self.aug = bool(config.get("aug", False))
        self.mmap = bool(config.get("mmap", False))

        spec = json.loads((self.root / "quant_spec.json").read_text())
        self.qmin = np.asarray(spec["qmin"], np.float32)
        self.qscale = np.asarray(spec["scale"], np.float32)
        # ALWAYS train statistics, whatever the zone (binding clause 2 of the
        # split contract). Schema: {"mean": [15 floats], "std": [15 floats]}.
        stats = json.loads((self.root / "train_stats.json").read_text())
        self.mean = np.asarray(stats["mean"], np.float32)[self.channels]
        self.std = np.asarray(stats["std"], np.float32)[self.channels]

        self.paths = sorted((self.root / zone).glob("shard_s*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"no shards under {self.root / zone}")
        limit = config.get("limit")  # --smoke: cap patch count per zone
        self.index = []              # (shard_i, item_i) per patch
        self.has_debris = []         # per patch, for the oversampling weights
        self.pixel_stats = {"clean": 0, "debris": 0}  # for pos_weight_type: auto
        self._cache = {}
        for si, path in enumerate(self.paths):
            with np.load(path) as z:
                if self.mmap:
                    masks = z["masks"]
                else:
                    self._cache[si] = {k: z[k] for k in ("imgs", "masks", "dist", "names")}
                    masks = self._cache[si]["masks"]
            self.pixel_stats["clean"] += int(masks[..., 0].sum(dtype=np.int64))
            self.pixel_stats["debris"] += int(masks[..., 1].sum(dtype=np.int64))
            self.index += [(si, i) for i in range(masks.shape[0])]
            self.has_debris += masks[..., 1].any(axis=(1, 2)).tolist()
            if limit is not None and len(self.index) >= limit:
                self.index, self.has_debris = self.index[:limit], self.has_debris[:limit]
                break

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, ii = self.index[i]
        if self.mmap:
            with np.load(self.paths[si]) as z:
                q, m = z["imgs"][ii], z["masks"][ii]
                d, name = z["dist"][ii], str(z["names"][ii])
        else:
            s = self._cache[si]
            q, m, d, name = s["imgs"][ii], s["masks"][ii], s["dist"][ii], str(s["names"][ii])

        nodata = q == NODATA                     # (H,W,15), pre-standardisation
        valid = ~nodata.any(axis=-1)             # finite in ALL 15 channels (binding clause 1)
        img = self.qmin + q.astype(np.float32) * self.qscale   # dequantise
        img = img[..., self.channels]
        img = (img - self.mean) / self.std       # train stats for every zone
        img[nodata[..., self.channels]] = 0.0    # zero-fill; loss/metrics gate on `valid`

        glacier = (m.sum(-1) > 0).astype(np.float32)  # clean OR debris
        typ = m[..., 1].astype(np.float32)            # debris; meaningful only where glacier==1
        dist = d.astype(np.float32) / 32.0            # int8 [-32,31] -> [-1, ~0.97]

        if self.train:
            img, glacier, typ, dist, valid = self._augment(img, glacier, typ, dist, valid)
        return {
            "img": torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
            "glacier": torch.from_numpy(np.ascontiguousarray(glacier)),
            "type": torch.from_numpy(np.ascontiguousarray(typ)),
            "dist": torch.from_numpy(np.ascontiguousarray(dist)),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),
            "name": name,
        }

    def _augment(self, img, glacier, typ, dist, valid):
        c = self.crop
        y, x = (np.random.randint(0, s - c + 1) for s in img.shape[:2])
        arrs = [a[y:y + c, x:x + c] for a in (img, glacier, typ, dist, valid)]
        if np.random.rand() < 0.5:
            arrs = [np.flip(a, 0) for a in arrs]
        if np.random.rand() < 0.5:
            arrs = [np.flip(a, 1) for a in arrs]
        k = np.random.randint(4)
        arrs = [np.rot90(a, k) for a in arrs]
        img, glacier, typ, dist, valid = arrs
        if self.aug:  # per-band spectral jitter on the standardised values
            gain = np.random.normal(1.0, 0.02, img.shape[-1]).astype(np.float32)
            bias = np.random.normal(0.0, 0.02, img.shape[-1]).astype(np.float32)
            img = img * gain + bias
            img[~valid] = 0.0  # keep nodata pixels at zero
        return img, glacier, typ, dist, valid


def make_sampler(dataset, k):
    """WeightedRandomSampler with patch weight 1 + k*has_debris (config oversample_debris)."""
    w = 1.0 + float(k) * np.asarray(dataset.has_debris, dtype=np.float64)
    return WeightedRandomSampler(torch.as_tensor(w), num_samples=len(dataset), replacement=True)
