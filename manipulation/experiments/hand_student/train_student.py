#!/usr/bin/env python3
"""Train the student on the cached 128x128 frames.

    python train_student.py CACHE.npz OUT_DIR [--holdout-clip IMG_0030] [--epochs 20] [--tag name]

Validation = the last 20% (in time) of every training clip; --holdout-clip additionally keeps one whole clip as a TEST set
(a different take), which measures how well the student generalises beyond the frames it was trained on.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from armcopy.nets import HandNet
from hc_common import NAMES

ap = argparse.ArgumentParser()
ap.add_argument("cache", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--holdout-clip", default=None)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--tag", default="student")
ap.add_argument("--grip-weight", type=float, default=3.0, help="extra loss weight on the open/closed output")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device("cuda")

d = np.load(args.cache)
X, Y, cid, fid, names = d["images"], d["targets"], d["clip_id"], d["frame_id"], list(d["clip_names"])
tail = np.zeros(len(X), bool)                                   # last 20% (in time) of each clip -> validation
for c in range(len(names)):
    idx = np.flatnonzero(cid == c)
    tail[idx[int(len(idx) * 0.8):]] = True
test = np.zeros(len(X), bool)
if args.holdout_clip:
    test = cid == names.index(args.holdout_clip)
train = ~tail & ~test
val = tail & ~test
print(f"clips {names}\ntrain {train.sum()}  val {val.sum()}  test {test.sum()} ({args.holdout_clip})")

mean, std = Y[train].mean(0), Y[train].std(0) + 1e-6
Xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()       # uint8 NCHW on CPU, moved per batch
Yt = torch.from_numpy(Y)
model = HandNet().to(dev)
model.t_mean.copy_(torch.from_numpy(mean)); model.t_std.copy_(torch.from_numpy(std))
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
steps = args.epochs * int(np.ceil(train.sum() / args.batch))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
scaler = torch.amp.GradScaler()
tm, ts = torch.from_numpy(mean).to(dev), torch.from_numpy(std).to(dev)


def augment(x, y):
    """x float 0..255 NCHW on GPU, y real-unit targets. Photometric + tiny shifts (labels shifted to match)."""
    n = x.shape[0]
    x = x * torch.empty(n, 1, 1, 1, device=dev).uniform_(0.6, 1.4)                               # brightness
    gain = torch.empty(n, 3, 1, 1, device=dev).uniform_(0.88, 1.12); x = x * gain                # colour cast
    m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * torch.empty(n, 1, 1, 1, device=dev).uniform_(0.7, 1.3) + m   # contrast
    if np.random.rand() < 0.5:                                                                    # blur (motion/defocus)
        k = torch.tensor([1., 2., 1.], device=dev); k = (k[:, None] * k[None, :]); k = (k / k.sum()).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k, groups=3)
    x = x + torch.randn_like(x) * torch.empty(n, 1, 1, 1, device=dev).uniform_(0, 8)             # sensor noise
    shift = torch.empty(n, 2, device=dev).uniform_(-0.05, 0.05) * x.shape[-1]                      # up to 5% of the image
    theta = torch.zeros(n, 2, 3, device=dev); theta[:, 0, 0] = 1; theta[:, 1, 1] = 1
    S = x.shape[-1]
    theta[:, :, 2] = -2 * shift / S                                                                # content moves by +shift
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    y = y.clone(); y[:, 0] += shift[:, 0] / S; y[:, 1] += shift[:, 1] / S
    return x.clamp(0, 255), y


@torch.no_grad()
def predict(mask):
    model.eval(); out = []
    idx = np.flatnonzero(mask)
    for i in range(0, len(idx), 256):
        out.append(model(Xt[idx[i:i + 256]].to(dev).float()).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 4)), idx


def report(mask, label):
    p, idx = predict(mask)
    if not len(idx):
        return None
    err = np.abs(p - Y[idx])
    r = {"n": int(len(idx)), "mae": {n: round(float(err[:, k].mean()), 4) for k, n in enumerate(NAMES)},
         "pos_err_pct_of_frame": round(float(np.hypot(*(err[:, :2].T * 100)).mean()), 2),
         "grip_correct_frac": round(float(((p[:, 3] > 0.5) == (Y[idx, 3] > 0.5)).mean()), 3),
         "size_rel_err": round(float((err[:, 2] / np.maximum(Y[idx, 2], 1e-3)).mean()), 3)}
    print(f"  {label}: {json.dumps(r)}")
    return r


train_idx = np.flatnonzero(train)
t0 = time.time()
for ep in range(args.epochs):
    model.train(); np.random.shuffle(train_idx); tot = 0.0
    for i in range(0, len(train_idx), args.batch):
        b = train_idx[i:i + args.batch]
        if len(b) < 8: continue
        x = Xt[b].to(dev).float(); y = Yt[b].to(dev)
        x, y = augment(x, y)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model.forward_standardised(x)
        w = torch.tensor([1.0, 1.0, 1.0, args.grip_weight], device=dev)
        loss = (F.smooth_l1_loss(pred.float(), (y - tm) / ts, reduction='none') * w).mean()
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += float(loss) * len(b)
    if ep % 4 == 3 or ep == args.epochs - 1:
        print(f"epoch {ep + 1}/{args.epochs}  train loss {tot / len(train_idx):.4f}  ({time.time() - t0:.0f}s)")
        report(val, "val (last 20% of each clip)")
final = {"val": report(val, "FINAL val"), "test": report(test, f"FINAL test on whole clip {args.holdout_clip}") if args.holdout_clip else None}
torch.save({"state_dict": model.state_dict(), "target_mean": mean, "target_std": std}, args.out / f"{args.tag}.pt")
for name, mask in (("val", val), ("test", test)):
    if mask.any():
        p, idx = predict(mask)
        np.savez_compressed(args.out / f"{args.tag}_pred_{name}.npz", pred=p, target=Y[idx], clip=cid[idx], frame=fid[idx], clip_names=np.array(names))
(args.out / f"{args.tag}_metrics.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, **final}, indent=1))
print("saved", args.out / f"{args.tag}.pt")
