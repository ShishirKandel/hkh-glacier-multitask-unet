"""Training entry point: python train.py --config configs/a.yaml [--smoke N].

Trains on zone train/, selects on zone val/ with a smoothed criterion.
test/ and east/ are NEVER opened here -- they belong to eval.py alone
(binding clauses 3-4 of the split contract).
"""
import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import ShardDataset, make_sampler
from losses import compute_loss
from model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # deterministic where feasible
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32  # distinct, reproducible per-worker seed
    np.random.seed(s)
    random.seed(s)


@torch.no_grad()
def validate(model, loader, device):
    """Masked val metrics. Debris prediction is the composed class
    (glacier>0.5 AND type>0.5), matching eval.py's 3-class composition."""
    model.eval()
    g_i = g_u = d_tp = d_fp = d_fn = 0
    for batch in loader:
        valid = batch["valid"].to(device)
        out = model(batch["img"].to(device))
        pg = (torch.sigmoid(out["glacier"][:, 0]) > 0.5) & valid
        pd = pg & (torch.sigmoid(out["type"][:, 0]) > 0.5)
        tg = (batch["glacier"].to(device) > 0.5) & valid
        td = (batch["type"].to(device) > 0.5) & valid
        g_i += (pg & tg).sum().item()
        g_u += (pg | tg).sum().item()
        d_tp += (pd & td).sum().item()
        d_fp += (pd & ~td).sum().item()
        d_fn += (~pd & td).sum().item()
    return {"glacier_iou": g_i / max(g_u, 1),
            "debris_iou": d_tp / max(d_tp + d_fp + d_fn, 1),
            "debris_precision": d_tp / max(d_tp + d_fp, 1),
            "debris_recall": d_tp / max(d_tp + d_fn, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", type=int, default=None,
                    help="first N patches per zone, 2 epochs, tiny model (CPU sanity run)")
    ap.add_argument("--data", default="shards",
                    help="shard root: train/ val/ + quant_spec.json + train_stats.json")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run = cfg.get("run", Path(args.config).stem)
    if args.smoke:
        cfg.update(limit=args.smoke, epochs=2, crop=128)
        cfg.setdefault("model", {})["base"] = 8
        run += "_smoke"
    out = Path(args.out) / run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))  # effective config

    set_seed(int(cfg.get("seed", 20260716)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ShardDataset(args.data, "train", cfg, train=True)
    val_ds = ShardDataset(args.data, "val", cfg, train=False)
    k = int(cfg.get("oversample_debris", 0))
    sampler = make_sampler(train_ds, k) if k > 0 else None
    batch = min(int(cfg.get("batch", 8)), len(train_ds))
    nw = int(cfg.get("num_workers", 2))
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=batch, sampler=sampler,
                          shuffle=sampler is None, drop_last=True, num_workers=nw,
                          worker_init_fn=seed_worker, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=nw, pin_memory=pin)

    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    use_amp = device.type == "cuda"  # AMP autocast only when CUDA is available
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    pw_cfg = cfg["loss"].get("pos_weight_type", 1.0)
    if pw_cfg == "auto":  # inverse debris frequency among TRAIN glacier pixels (clause 2)
        pw = train_ds.pixel_stats["clean"] / max(train_ds.pixel_stats["debris"], 1)
    else:
        pw = float(pw_cfg)
    pos_weight = torch.tensor(pw, device=device)

    mfile = (out / "metrics.jsonl").open("w")
    mfile.write(json.dumps({"versions": {"torch": torch.__version__, "numpy": np.__version__},
                            "device": str(device), "pos_weight_type": round(pw, 4),
                            "n_train": len(train_ds), "n_val": len(val_ds)}) + "\n")

    # Model selection (binding clause 4): val holds only 118 debris patches, so a single
    # debris-IoU evaluation carries ~4-5 pp standard error -- a raw per-epoch argmax would
    # mostly select noise. We therefore score each epoch as mean(glacier IoU, debris IoU)
    # averaged over the LAST 3 val evaluations, and checkpoint the best smoothed score.
    window = deque(maxlen=3)
    best, since_best = -1.0, 0
    for epoch in range(1, int(cfg.get("epochs", 30)) + 1):
        model.train()
        t0, sums, nb = time.time(), {}, 0
        for b in train_dl:
            b = {key: (v.to(device) if torch.is_tensor(v) else v) for key, v in b.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                total, parts = compute_loss(model(b["img"]), b, cfg["loss"], pos_weight)
            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()
            nb += 1
            parts["total"] = float(total.detach())
            for key, v in parts.items():
                sums[key] = sums.get(key, 0.0) + v
        train_log = {key: round(v / max(nb, 1), 5) for key, v in sums.items()}

        vm = validate(model, val_dl, device)
        window.append(0.5 * (vm["glacier_iou"] + vm["debris_iou"]))
        smoothed = sum(window) / len(window)
        ckpt = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "score": smoothed}
        torch.save(ckpt, out / "last.pt")
        # Only compete for "best" once the smoothing window is full: a lucky epoch-1
        # spike is a length-1 window and would otherwise set an unbeatable score
        # (observed: rung b1 froze on epoch 1 and early-stopped mid-climb at epoch 9).
        if len(window) == window.maxlen and smoothed > best:
            best, since_best = smoothed, 0
            torch.save(ckpt, out / "best.pt")
        else:
            since_best += 1
        mfile.write(json.dumps({"epoch": epoch, "train": train_log, "val": vm,
                                "score_smoothed": smoothed,
                                "time_s": round(time.time() - t0, 1)}) + "\n")
        mfile.flush()
        print(f"epoch {epoch:3d}  loss={train_log['total']:.4f}  "
              f"gIoU={vm['glacier_iou']:.4f}  dIoU={vm['debris_iou']:.4f}  "
              f"smoothed={smoothed:.4f}")
        # min_epochs guards against stopping before the slow climb out of the
        # prior-initialised all-negative start (rung a first crossed IoU 0.3 at ep 10+).
        if epoch >= int(cfg.get("min_epochs", 15)) and since_best >= int(cfg.get("patience", 8)):
            print(f"early stop: no smoothed-score improvement for {since_best} epochs")
            break
    mfile.close()
    print(f"done: best smoothed score {best:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
