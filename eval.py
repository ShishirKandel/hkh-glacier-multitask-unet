"""One-shot evaluation of a trained checkpoint on val/test/east shards.

Split-contract reminder: test/ is evaluated exactly once, after model selection
on val/; east/ is out-of-distribution reporting only -- never for selection.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ShardDataset
from model import build_model

CLASSES = ("background", "clean", "debris")


@torch.no_grad()
def evaluate(model, loader, device, n_examples=0):
    """Full-resolution masked evaluation. Returns (3x3 confusion matrix over valid
    pixels, dist MAE in pixels or None, first-K example arrays for PNG export)."""
    model.eval()
    cm = np.zeros((3, 3), np.int64)  # rows = truth, cols = prediction
    mae_sum, mae_n = 0.0, 0
    examples = []
    for batch in loader:
        img = batch["img"].to(device)
        valid = batch["valid"].to(device)
        out = model(img)
        # 3-class composition: glacier<0.5 -> background, else type<0.5 -> clean else debris
        pg = torch.sigmoid(out["glacier"][:, 0]) > 0.5
        pt = torch.sigmoid(out["type"][:, 0]) > 0.5
        pred = pg.long() + (pg & pt).long()
        tg = batch["glacier"].to(device) > 0.5   # union label, so debris implies glacier
        td = batch["type"].to(device) > 0.5
        truth = tg.long() + td.long()
        v = valid.reshape(-1)
        idx = (truth.reshape(-1)[v] * 3 + pred.reshape(-1)[v]).cpu().numpy()
        cm += np.bincount(idx, minlength=9).reshape(3, 3)
        if "dist" in out:
            m = valid.float()
            mae_sum += ((out["dist"][:, 0] - batch["dist"].to(device)).abs() * m).sum().item()
            mae_n += m.sum().item()
        for j in range(img.shape[0]):
            if len(examples) >= n_examples:
                break
            rgb = img[j, :3] if img.shape[1] >= 3 else img[j, :1].repeat(3, 1, 1)
            examples.append({"rgb": rgb.cpu().numpy().transpose(1, 2, 0),
                             "truth": truth[j].cpu().numpy(),
                             "pred": pred[j].cpu().numpy(),
                             "name": batch["name"][j]})
    mae_px = 32.0 * mae_sum / mae_n if mae_n else None  # undo the /32 target scaling
    return cm, mae_px, examples


def binary_metrics(tp, fp, fn):
    tp, fp, fn = float(tp), float(fp), float(fn)
    return {"iou": tp / max(tp + fp + fn, 1.0),
            "dice": 2 * tp / max(2 * tp + fp + fn, 1.0),
            "precision": tp / max(tp + fp, 1.0),
            "recall": tp / max(tp + fn, 1.0)}


def metrics_from_cm(cm):
    res = {}
    # glacier head as a binary problem: classes {clean, debris} vs background
    res["glacier"] = binary_metrics(cm[1:, 1:].sum(), cm[0, 1:].sum(), cm[1:, 0].sum())
    f1s = []
    for c, name in enumerate(CLASSES):
        m = binary_metrics(cm[c, c], cm[:, c].sum() - cm[c, c], cm[c, :].sum() - cm[c, c])
        f1s.append(m["dice"])  # Dice == F1 per class
        if name != "background":
            res[name] = m
    res["macro_f1"] = float(np.mean(f1s))
    return res


def save_examples(examples, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["black", "#4da6ff", "#ff8c42"])  # background, clean, debris
    out_dir.mkdir(parents=True, exist_ok=True)
    for ex in examples:
        lo, hi = np.percentile(ex["rgb"], [2, 98])
        rgb = np.clip((ex["rgb"] - lo) / max(hi - lo, 1e-6), 0, 1)
        fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
        ax[0].imshow(rgb)
        ax[1].imshow(ex["truth"], cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        ax[2].imshow(ex["pred"], cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        for a, title in zip(ax, ("input (first 3 bands)", "ground truth", "prediction")):
            a.set_title(title)
            a.axis("off")
        fig.suptitle(ex["name"], fontsize=9)
        fig.savefig(out_dir / f"{ex['name']}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--zone", required=True, choices=["val", "test", "east"])
    ap.add_argument("--data", default="shards")
    ap.add_argument("--examples", type=int, default=None,
                    help="override config eval_examples (K prediction PNGs)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    ds = ShardDataset(args.data, args.zone, cfg, train=False)  # full 512, no aug
    dl = DataLoader(ds, batch_size=4, shuffle=False,
                    num_workers=int(cfg.get("num_workers", 2)))
    n_ex = args.examples if args.examples is not None else int(cfg.get("eval_examples", 0))
    cm, mae_px, examples = evaluate(model, dl, device, n_ex)

    res = metrics_from_cm(cm)
    res.update(confusion_matrix=cm.tolist(), zone=args.zone, ckpt=str(args.ckpt),
               epoch=ckpt.get("epoch"), n_patches=len(ds))
    if mae_px is not None:
        res["dist_mae_px"] = mae_px
    out_dir = Path(args.ckpt).parent
    (out_dir / f"eval_{args.zone}.json").write_text(json.dumps(res, indent=1))
    if examples:
        save_examples(examples, out_dir / f"examples_{args.zone}")

    g, d = res["glacier"], res["debris"]
    print(f"{args.zone}: gIoU={g['iou']:.4f} gDice={g['dice']:.4f} "
          f"dIoU={d['iou']:.4f} dRec={d['recall']:.4f} dPrec={d['precision']:.4f} "
          f"macroF1={res['macro_f1']:.4f}"
          + (f" distMAE={mae_px:.2f}px" if mae_px is not None else "")
          + f" n={len(ds)}")


if __name__ == "__main__":
    main()
