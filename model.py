"""U-Net from scratch (no torchvision/smp) with optional additive attention gates
on the skip connections and multi-task heads (glacier, type, optional dist) as
separate 1x1 convolutions on the shared final decoder feature map.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 conv + BatchNorm + ReLU, optional Dropout2d at the end."""

    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        layers = []
        for c in (cin, cout):
            layers += [nn.Conv2d(c, cout, 3, padding=1, bias=False),
                       nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018): skip features x are re-weighted
    by a sigmoid coefficient map computed from x and the (already upsampled, so
    same-resolution) decoder gating signal g."""

    def __init__(self, ch, mid):
        super().__init__()
        self.wg = nn.Conv2d(ch, mid, 1, bias=False)
        self.wx = nn.Conv2d(ch, mid, 1, bias=False)
        self.psi = nn.Conv2d(mid, 1, 1)

    def forward(self, g, x):
        a = torch.sigmoid(self.psi(F.relu(self.wg(g) + self.wx(x))))
        return x * a


class UNet(nn.Module):
    def __init__(self, in_ch, depth=4, base=32, dropout=0.0, attention=False,
                 dist_head=False, priors=None):
        super().__init__()
        chans = [base * 2 ** i for i in range(depth + 1)]  # encoder widths + bottleneck
        self.pool = nn.MaxPool2d(2)
        self.enc = nn.ModuleList()
        c = in_ch
        for co in chans[:-1]:
            self.enc.append(ConvBlock(c, co, dropout))
            c = co
        self.bottleneck = ConvBlock(chans[-2], chans[-1], dropout)
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        self.att = nn.ModuleList() if attention else None
        for hi, lo in zip(chans[:0:-1], chans[-2::-1]):  # e.g. 512->256, ..., 64->32
            self.up.append(nn.ConvTranspose2d(hi, lo, 2, stride=2))
            self.dec.append(ConvBlock(2 * lo, lo, dropout))
            if attention:
                self.att.append(AttentionGate(lo, max(lo // 2, 1)))
        # Heads: glacier-vs-background logit, debris-vs-clean logit (within glacier),
        # optional linear dist regression. Registered with a "head_" prefix because
        # nn.ModuleDict rejects the key "type" (it shadows nn.Module.type()).
        self.head_names = ["glacier", "type"] + (["dist"] if dist_head else [])
        for name in self.head_names:
            self.add_module("head_" + name, nn.Conv2d(base, 1, 1))
        # Prior-logit bias init (RetinaNet-style): start each classification head at
        # its class prior instead of p=0.5. Without this, unweighted BCE on ~7%-positive
        # crops drives the net into the all-background attractor within ~2 epochs
        # (observed: val IoU -> 0 by epoch 4). This is initialisation hygiene applied
        # to every rung; class-imbalance LOSS handling still only enters at rung (d).
        for name, p in (priors or {}).items():
            if name in self.head_names and name != "dist":
                b = getattr(self, "head_" + name).bias
                nn.init.constant_(b, float(torch.log(torch.tensor(p / (1.0 - p)))))

    def forward(self, x):
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x)
            s = skips[-(i + 1)]
            if self.att is not None:
                s = self.att[i](x, s)
            x = dec(torch.cat([x, s], dim=1))
        return {name: getattr(self, "head_" + name)(x) for name in self.head_names}


def build_model(cfg):
    """Build a UNet from a full run config (channels, model.*, heads.dist)."""
    m = cfg.get("model", {})
    return UNet(in_ch=len(cfg["channels"]),
                depth=int(m.get("depth", 4)),
                base=int(m.get("base", 32)),
                dropout=float(m.get("dropout", 0.0)),
                attention=bool(m.get("attention", False)),
                dist_head=bool(cfg.get("heads", {}).get("dist", False)),
                priors=m.get("priors", {"glacier": 0.08, "type": 0.10}))
