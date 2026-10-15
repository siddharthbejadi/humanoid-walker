#!/usr/bin/env python3
"""Gripper network: hand crop -> grip in [0,1] (0 open hand, 1 fist), trained on FreiHAND (non-commercial research data).
    python train_grip.py FREIHAND_DIR OUT_DIR [--epochs 12]
Label = finger curl from the 3D joints (same definition as hc_common.targets_from_landmarks: tip-wrist / MCP-wrist, percentile-calibrated).
Augmentation mimics a hand seen from 2 m on a board camera: random crop/scale, down-up scaling (blur), colour casts (red sleeves), noise."""
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision

ap = argparse.ArgumentParser(); ap.add_argument("data", type=Path); ap.add_argument("out", type=Path)
ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--size", type=int, default=96); ap.add_argument("--tag", default="grip_v1")
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda"); torch.manual_seed(0); np.random.seed(0)

xyz = np.array(json.load(open(a.data / "training_xyz.json")), np.float32)          # (32560, 21, 3) metres
ids = json.load(open(a.data / "ids.json"))
TIPS, MCPS = [8, 12, 16, 20], [5, 9, 13, 17]
def openness(j): return float(np.mean(np.linalg.norm(j[TIPS] - j[0], axis=1) / np.linalg.norm(j[MCPS] - j[0], axis=1)))
op = np.array([openness(x) for x in xyz]); lo, hi = np.percentile(op, [5, 95])          # calibrate: 5th pct = fist (1), 95th pct = open (0)
print(f"openness percentiles 5/50/95: {lo:.2f} {np.median(op):.2f} {hi:.2f}")
grip_all = np.clip((hi - op) / (hi - lo), 0, 1).astype(np.float32)

imgs, y = [], []
for i in ids:
    p = a.data / "rgb" / f"{i:08d}.jpg"
    if p.exists(): imgs.append(cv2.resize(cv2.imread(str(p)), (128, 128), interpolation=cv2.INTER_AREA)[:, :, ::-1]); y.append(grip_all[i % 32560])
X = np.stack(imgs); y = np.array(y, np.float32); n = len(X); ntr = int(n * 0.85)
print(f"{n} images; grip distribution: mean {y.mean():.2f}, fist(>0.7) {100*np.mean(y>0.7):.0f}%, open(<0.3) {100*np.mean(y<0.3):.0f}%")
Xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous(); yt = torch.from_numpy(y)

net = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
class Grip(nn.Module):
    def __init__(s):
        super().__init__(); s.f = net.features; s.h = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.1), nn.Linear(1280, 1))
        s.register_buffer("m", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255); s.register_buffer("s", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)
    def forward(s, x): return s.h(s.f((x - s.m) / s.s))[:, 0]                          # logit; grip = sigmoid(logit)
model = Grip().to(dev); S = a.size
opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4); bs = 128
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=6e-4, total_steps=a.epochs * (ntr // bs + 1), pct_start=0.15); scaler = torch.amp.GradScaler()

def augment(x):                                                   # x uint8->float 0..255 (B,3,128,128) on GPU
    B = x.shape[0]; sc = torch.empty(B, device=dev).uniform_(0.62, 1.0); sh = torch.empty(B, 2, device=dev).uniform_(-0.12, 0.12)
    th = torch.zeros(B, 2, 3, device=dev); th[:, 0, 0] = sc; th[:, 1, 1] = sc; th[:, :, 2] = sh * (1 - sc[:, None] + 0.1)
    rot = torch.empty(B, device=dev).uniform_(-0.35, 0.35); c, s_ = torch.cos(rot), torch.sin(rot)
    th[:, 0, 0] = sc * c; th[:, 0, 1] = -sc * s_; th[:, 1, 0] = sc * s_; th[:, 1, 1] = sc * c
    x = F.grid_sample(x, F.affine_grid(th, (B, 3, S, S), align_corners=False), padding_mode="reflection", align_corners=False)
    small = int(np.random.choice([18, 24, 32, 48, 96]))             # hand far away -> tiny -> blown back up
    x = F.interpolate(F.interpolate(x, size=(small, small), mode="area"), size=(S, S), mode="bilinear", align_corners=False)
    x = x * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.6, 1.4) * torch.empty(B, 3, 1, 1, device=dev).uniform_(0.8, 1.25)
    x = x + torch.randn_like(x) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0, 10)
    return x.clamp(0, 255)
def prep_eval(x): return F.interpolate(x, size=(S, S), mode="area")

@torch.no_grad()
def evaluate(idx, far=False):
    model.eval(); p = []
    for i in range(0, len(idx), 256):
        x = Xt[idx[i:i + 256]].to(dev).float(); x = prep_eval(x)
        if far: x = F.interpolate(F.interpolate(x, size=(24, 24), mode="area"), size=(S, S), mode="bilinear", align_corners=False)
        p.append(torch.sigmoid(model(x)).cpu())
    p = torch.cat(p).numpy(); t = y[idx]
    return {"mae": round(float(np.abs(p - t).mean()), 3), "open_closed_acc": round(float(((p > 0.5) == (t > 0.5)).mean()), 3), "baseline_acc_always_majority": round(float(max((t > 0.5).mean(), (t <= 0.5).mean())), 3)}
val = np.arange(ntr, n); tr = np.arange(ntr); t0 = time.time()
for ep in range(a.epochs):
    model.train(); np.random.shuffle(tr); tot = 0
    for i in range(0, len(tr) - bs + 1, bs):
        b = tr[i:i + bs]; x = augment(Xt[b].to(dev).float()); t = yt[b].to(dev)
        with torch.autocast("cuda", dtype=torch.float16): lg = model(x)
        loss = F.binary_cross_entropy_with_logits(lg.float(), t); opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tot += float(loss.detach())
    if ep % 3 == 2 or ep == a.epochs - 1: print(f"epoch {ep + 1}/{a.epochs} loss {tot / (len(tr) // bs):.4f} ({time.time() - t0:.0f}s) val {evaluate(val)} val-far {evaluate(val, far=True)}", flush=True)
res = {"val_close": evaluate(val), "val_far_24px": evaluate(val, far=True), "n_train": int(ntr), "n_val": int(n - ntr), "openness_p5_p95": [float(lo), float(hi)]}
torch.save({"state_dict": model.state_dict(), "size": S, "openness_p5_p95": [float(lo), float(hi)]}, a.out / f"{a.tag}.pt"); (a.out / f"{a.tag}_metrics.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res))
dummy = torch.zeros(1, 3, S, S, device=dev); model.eval(); torch.onnx.export(model, dummy, str(a.out / f"{a.tag}.onnx"), input_names=["image"], output_names=["logit"], opset_version=13, dynamo=False); print("exported onnx")
