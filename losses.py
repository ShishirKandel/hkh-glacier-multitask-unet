"""Masked per-head losses. Every term is gated by the nodata `valid` mask
(binding clause 1); the type head is additionally masked to glacier==1 pixels
on the label side, so it only learns debris-vs-clean within true glacier.
"""
import torch
import torch.nn.functional as F


def _mean(loss, mask):
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def masked_bce(logits, target, mask, pos_weight=None):
    l = F.binary_cross_entropy_with_logits(logits, target, reduction="none",
                                           pos_weight=pos_weight)
    return _mean(l, mask)


def masked_dice(logits, target, mask, eps=1.0):
    """Soft Dice on sigmoid probabilities, pooled over the whole (masked) batch."""
    p, t = torch.sigmoid(logits) * mask, target * mask
    return 1.0 - (2.0 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)


def masked_focal(logits, target, mask, gamma=2.0, pos_weight=None):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none",
                                             pos_weight=pos_weight)
    p = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, p, 1.0 - p)  # probability of the true class
    return _mean((1.0 - pt) ** gamma * bce, mask)


def masked_l1(pred, target, mask):
    return _mean((pred - target).abs(), mask)


def compute_loss(outputs, batch, loss_cfg, pos_weight=None):
    """Compound loss: weighted sum of terms per head, then head_weights across heads.

    loss_cfg example: {glacier: {bce: 1, dice: 1}, type: {focal: 1}, dist: {l1: 1},
    head_weights: {glacier: 1, type: 1.5, dist: 0.5}, focal_gamma: 2.0}.
    pos_weight (clean/debris pixel ratio, or a config constant) applies to the
    type head only. Returns (total, {component_name: float}).
    """
    valid = batch["valid"].float()
    head_mask = {"glacier": valid,
                 "type": valid * batch["glacier"],  # only true-glacier pixels
                 "dist": valid}
    target = {"glacier": batch["glacier"], "type": batch["type"], "dist": batch["dist"]}
    hw = loss_cfg.get("head_weights", {})
    gamma = float(loss_cfg.get("focal_gamma", 2.0))
    total, parts = 0.0, {}
    for head in ("glacier", "type", "dist"):
        terms = loss_cfg.get(head)
        if terms is None or head not in outputs:
            continue
        logits, m, t = outputs[head][:, 0], head_mask[head], target[head]
        pw = pos_weight if head == "type" else None
        for term, w in terms.items():
            if term == "bce":
                l = masked_bce(logits, t, m, pw)
            elif term == "dice":
                l = masked_dice(logits, t, m)
            elif term == "focal":
                l = masked_focal(logits, t, m, gamma, pw)
            elif term == "l1":
                l = masked_l1(logits, t, m)  # dist head: raw linear output
            else:
                raise ValueError(f"unknown loss term {term!r} for head {head!r}")
            parts[f"{head}_{term}"] = float(l.detach())
            total = total + float(hw.get(head, 1.0)) * float(w) * l
    return total, parts
