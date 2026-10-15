#!/usr/bin/env python3
"""Train the joint-space student: camera image (192x192) -> [shoulder, arm, elbow, present-logit].

    ~/hc_train/bin/python -I train_arm_student.py CACHE.npz OUT_DIR [--holdout-clip NAME] [--epochs 30] [--tag name]

Labels come from the teacher (MoveNet + IK, see label_arm.py).  Regression loss only on frames where the teacher found a plausible arm;
'present' (BCE) on every frame, so the student also learns to say "I can't see an arm" (the live demo then holds/relaxes).
Validation = last 20% in time of every clip with an arm; --holdout-clip keeps a whole clip as an unseen test."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hc_model import HandNet

ap = argparse.ArgumentParser()
ap.add_argument("cache", type=Path); ap.add_argument("out", type=Path)
ap.add_argument("--holdout-clip", default=None); ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--final-clip", default=None, help="the deployment view: oversampled x3 and reported separately")
ap.add_argument("--tag", default="armnet"); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(args.seed); np.random.seed(args.seed); dev = torch.device("cuda")

d = np.load(args.cache)
X, Q, valid, cid, fid, names = d["images"], d["q"], d["valid"], d["clip_id"], d["frame_id"], list(d["clip_names"])
pos_clips = [c for c in range(len(names)) if valid[cid == c].any()]
tail = np.zeros(len(X), bool)
for c in pos_clips:
    idx = np.flatnonzero(cid == c); tail[idx[int(len(idx) * 0.8):]] = True
neg_tail = np.zeros(len(X), bool)
for c in range(len(names)):
    if c not in pos_clips:
        idx = np.flatnonzero(cid == c); neg_tail[idx[int(len(idx) * 0.8):]] = True
test = (cid == names.index(args.holdout_clip)) if args.holdout_clip else np.zeros(len(X), bool)
val = (tail | neg_tail) & ~test; train = ~(tail | neg_tail) & ~test
print(f"clips {names}\ntrain {train.sum()} (arm {valid[train].sum()})  val {val.sum()} (arm {valid[val].sum()})  test {test.sum()} (arm {valid[test].sum()})")

Y = np.concatenate([Q, valid[:, None].astype(np.float32)], 1)
mean = np.concatenate([Q[train & valid].mean(0), [0.0]]).astype(np.float32); std = np.concatenate([Q[train & valid].std(0) + 1e-3, [1.0]]).astype(np.float32)
Xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous(); Yt = torch.from_numpy(Y)
model = HandNet().to(dev); model.t_mean.copy_(torch.from_numpy(mean)); model.t_std.copy_(torch.from_numpy(std))
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
n_eff = train.sum() + (2 * (train & (cid == names.index(args.final_clip))).sum() if args.final_clip else 0)   # take-3 frames repeat x3
steps = args.epochs * int(np.ceil(n_eff / args.batch)) + 5
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
scaler = torch.amp.GradScaler(); tm, ts = torch.from_numpy(mean).to(dev), torch.from_numpy(std).to(dev)
# positives are rarer than negatives in the cache: balance present-loss
pos_w = float((~valid[train]).sum() / max(valid[train].sum(), 1)); pos_w = min(pos_w, 4.0)


def augment(x):
    n = x.shape[0]
    x = x * torch.empty(n, 1, 1, 1, device=dev).uniform_(0.55, 1.45)
    x = x * torch.empty(n, 3, 1, 1, device=dev).uniform_(0.88, 1.12)
    m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * torch.empty(n, 1, 1, 1, device=dev).uniform_(0.7, 1.3) + m
    if np.random.rand() < 0.5:
        k = torch.tensor([1., 2., 1.], device=dev); k = (k[:, None] * k[None, :]); k = (k / k.sum()).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k, groups=3)
    x = x + torch.randn_like(x) * torch.empty(n, 1, 1, 1, device=dev).uniform_(0, 8)
    sc = torch.empty(n, device=dev).uniform_(0.92, 1.08); sh = torch.empty(n, 2, device=dev).uniform_(-0.06, 0.06)   # the arm DIRECTION is invariant to this
    theta = torch.zeros(n, 2, 3, device=dev); theta[:, 0, 0] = 1 / sc; theta[:, 1, 1] = 1 / sc; theta[:, :, 2] = sh
    return F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="zeros", align_corners=False).clamp(0, 255)


@torch.no_grad()
def predict(mask):
    model.eval(); idx = np.flatnonzero(mask); out = []
    for i in range(0, len(idx), 256):
        out.append(model(Xt[idx[i:i + 256]].to(dev).float()).cpu())
    return (torch.cat(out).numpy() if out else np.zeros((0, 4))), idx


def report(mask, label):
    p, idx = predict(mask)
    if not len(idx): return None
    v = valid[idx]; pres = p[:, 3] > 0
    r = {"n": int(len(idx)), "n_arm": int(v.sum()),
         "joint_mae_deg": [round(float(np.degrees(np.abs(p[v, k] - Q[idx][v, k]).mean())), 1) for k in range(3)] if v.any() else None,
         "baseline_mean_pose_mae_deg": [round(float(np.degrees(np.abs(mean[k] - Q[idx][v, k]).mean())), 1) for k in range(3)] if v.any() else None,
         "present_acc": round(float((pres == v).mean()), 3), "present_recall": round(float(pres[v].mean()), 3) if v.any() else None,
         "false_alarm_rate": round(float(pres[~v].mean()), 3) if (~v).any() else None}
    print(f"  {label}: {json.dumps(r)}"); return r


train_idx = np.flatnonzero(train); t0 = time.time()
if args.final_clip:
    fi = train_idx[cid[train_idx] == names.index(args.final_clip)]; train_idx = np.concatenate([train_idx, fi, fi])
for ep in range(args.epochs):
    model.train(); np.random.shuffle(train_idx); tot = 0.0
    for i in range(0, len(train_idx), args.batch):
        b = train_idx[i:i + args.batch]
        if len(b) < 8: continue
        x = augment(Xt[b].to(dev).float()); y = Yt[b].to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            z = model.forward_standardised(x)
        z = z.float(); real = z * ts + tm
        m = y[:, 3:4]
        reg = (F.smooth_l1_loss(z[:, :3], (y[:, :3] - tm[:3]) / ts[:3], reduction="none") * m).sum() / (m.sum() * 3 + 1e-6)
        cls = F.binary_cross_entropy_with_logits(real[:, 3], y[:, 3], pos_weight=torch.tensor(1.0, device=dev))
        loss = reg + 0.5 * cls
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += float(loss.detach()) * len(b)
    if ep % 5 == 4 or ep == args.epochs - 1:
        print(f"epoch {ep + 1}/{args.epochs} loss {tot / len(train_idx):.4f} ({time.time() - t0:.0f}s)"); report(val, "val")
fmask = val & (cid == names.index(args.final_clip)) if args.final_clip else np.zeros(len(X), bool)
final = {"val": report(val, "FINAL val"), "final_clip_val": report(fmask, f"FINAL val on deployment view {args.final_clip}") if fmask.any() else None, "test": report(test, f"FINAL test {args.holdout_clip}") if args.holdout_clip else None}
torch.save({"state_dict": model.state_dict(), "target_mean": mean, "target_std": std}, args.out / f"{args.tag}.pt")
for name, mask in (("val", val), ("test", test), ("final", fmask)):
    if mask.any():
        p, idx = predict(mask)
        np.savez_compressed(args.out / f"{args.tag}_pred_{name}.npz", pred=p, q=Q[idx], valid=valid[idx], clip=cid[idx], frame=fid[idx], clip_names=np.array(names))
(args.out / f"{args.tag}_metrics.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, **final}, indent=1))
print("saved", args.out / f"{args.tag}.pt")
