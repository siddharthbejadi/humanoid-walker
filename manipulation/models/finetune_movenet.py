"""Fine-tune the ST model-zoo MoveNet (heatmaps, 192 px, Keras) on YOUR board-camera takes for the right arm  ->  the 'student'.

    python finetune_movenet.py CACHE.npz OUT.keras [--epochs 6]

Supervision (self-training with temporal cleaning): keypoints of the ORIGINAL zoo model on frames where the arm passed the plausibility gate
(label_arm.py 'valid'), smoothed over +-2 frames, turned into Gaussian target maps for the right-arm channels (5,6 shoulders, 8 elbow, 10 wrist).
All other channels are distilled from the frozen original so nothing else is forgotten.  Val = last 20% of take3 (never trained on).
Reports keypoint error, plausible-detection coverage and frame-to-frame jitter for original vs fine-tuned."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, tensorflow as tf, keras

ap = argparse.ArgumentParser()
ap.add_argument("cache", type=Path); ap.add_argument("out", type=Path)
ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--val-clip", default="take3")
a = ap.parse_args()
ZOO = Path(__file__).resolve().parent.parent / "models_zoo" / "st_movenet_192.keras"
ARM = [5, 6, 8, 10]; OTHER = [i for i in range(17) if i not in ARM]; G = 48
np.random.seed(0); tf.random.set_seed(0)

d = np.load(a.cache); X, valid, cid, fid, names = d["images"], d["valid"], d["clip_id"], d["frame_id"], list(d["clip_names"])
N = len(X); M0 = keras.models.load_model(ZOO, compile=False); M0.trainable = False
M = keras.models.load_model(ZOO, compile=False)
prep = lambda u8: tf.cast(u8, tf.float32) / 127.5 - 1.0


def softargmax(h, win=2):                                   # h (N,48,48,K) -> (N,K,3) [x, y, score] in cell units
    N_, _, _, K = h.shape; out = np.zeros((N_, K, 3), np.float32)
    for n in range(N_):
        for k in range(K):
            m = h[n, :, :, k]; y, x = np.unravel_index(m.argmax(), m.shape)
            y0, y1, x0, x1 = max(0, y - win), min(G, y + win + 1), max(0, x - win), min(G, x + win + 1)
            p = m[y0:y1, x0:x1]; s = p.sum() + 1e-9
            ys, xs = np.mgrid[y0:y1, x0:x1]
            out[n, k] = ((p * xs).sum() / s + 0.5, (p * ys).sum() / s + 0.5, m.max())
    return out


def predict_kp(model, imgs, ch):                            # keypoints of selected channels, batch by batch (heatmaps are too big to keep: 7 GB RAM)
    return np.concatenate([softargmax(model(prep(imgs[i:i + 64]), training=False).numpy()[..., ch]) for i in range(0, len(imgs), 64)])


t0 = time.time(); KP0 = predict_kp(M0, X, ARM); print(f"original model keypoints for {N} frames in {time.time() - t0:.0f}s", flush=True)
pos_clips = [c for c in range(len(names)) if valid[cid == c].any()]
# temporal smoothing (+-2 frames, valid neighbours only) -> training keypoints
KP = KP0.copy(); sup = np.zeros(N, bool)
for c in pos_clips:
    idx = np.flatnonzero(cid == c)
    for k in idx:
        if valid[k]:
            nb = [j for j in range(max(idx[0], k - 2), min(idx[-1], k + 2) + 1) if valid[j]]
            KP[k] = KP0[nb].mean(0); sup[k] = True
vc = names.index(a.val_clip); vidx = np.flatnonzero(cid == vc); val_mask = np.zeros(N, bool); val_mask[vidx[int(len(vidx) * 0.8):]] = True
train_mask = ~val_mask
print(f"train {train_mask.sum()} (arm-supervised {(sup & train_mask).sum()})  val {val_mask.sum()} (arm {(valid & val_mask).sum()})", flush=True)

yy, xx = np.mgrid[0:G, 0:G]
def target_maps(kp):                                        # (B,4,3) -> (B,48,48,4) normalised Gaussians (sigma 1.2 cells)
    g = np.exp(-(((xx[None, None] + 0.5 - kp[:, :, 0, None, None]) ** 2 + (yy[None, None] + 0.5 - kp[:, :, 1, None, None]) ** 2) / (2 * 1.2 ** 2)))
    g /= g.sum((2, 3), keepdims=True) + 1e-9
    return g.transpose(0, 2, 3, 1).astype(np.float32)

OTHER_T = tf.constant(OTHER); ARM_T = tf.constant(ARM)
def augment(x):                                             # x float 0..255
    n = tf.shape(x)[0]
    x = x * tf.random.uniform([n, 1, 1, 1], 0.6, 1.4) * tf.random.uniform([n, 1, 1, 3], 0.88, 1.12)
    m = tf.reduce_mean(x, [1, 2, 3], keepdims=True); x = (x - m) * tf.random.uniform([n, 1, 1, 1], 0.7, 1.3) + m
    x = x + tf.random.normal(tf.shape(x)) * tf.random.uniform([n, 1, 1, 1], 0, 8)
    return tf.clip_by_value(x, 0, 255)

opt = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(a.lr, a.epochs * (train_mask.sum() // a.batch), alpha=0.05))
@tf.function
def step(x, tgt, sup_b):
    h0 = M0(x / 127.5 - 1.0, training=False)
    with tf.GradientTape() as tape:
        p = M(x / 127.5 - 1.0, training=True)
        logp = tf.math.log(tf.clip_by_value(p, 1e-8, 1.0))
        arm_kl = -tf.reduce_sum(tgt * tf.gather(logp, ARM_T, axis=3), [1, 2])               # cross-entropy to the Gaussian targets (B,4)
        arm_loss = tf.reduce_sum(arm_kl * sup_b[:, None]) / (tf.reduce_sum(sup_b) * 4 + 1e-6)
        h0o = tf.gather(h0, OTHER_T, axis=3); po = tf.gather(p, OTHER_T, axis=3)
        dist = tf.reduce_mean(tf.reduce_sum(h0o * (tf.math.log(tf.clip_by_value(h0o, 1e-8, 1.0)) - tf.math.log(tf.clip_by_value(po, 1e-8, 1.0))), [1, 2]))
        loss = arm_loss + 0.3 * dist
    g = tape.gradient(loss, M.trainable_variables); opt.apply_gradients(zip(g, M.trainable_variables))
    return loss, arm_loss

tr = np.flatnonzero(train_mask)
for ep in range(a.epochs):
    np.random.shuffle(tr); tot = 0.0; nb = 0; t0 = time.time()
    for i in range(0, len(tr) - a.batch + 1, a.batch):
        b = np.sort(tr[i:i + a.batch])
        x = augment(tf.cast(X[b], tf.float32)); tgt = target_maps(KP[b]); sb = sup[b].astype(np.float32)
        loss, al = step(x, tgt, sb); tot += float(loss); nb += 1
    print(f"epoch {ep + 1}/{a.epochs} loss {tot / nb:.4f} ({time.time() - t0:.0f}s)", flush=True)
M.save(a.out)

# ---- evaluation on the unseen last 20% of take3: original vs fine-tuned ----
vi = np.flatnonzero(val_mask)
K1 = predict_kp(M, X[vi], ARM); K0 = KP0[vi]
def gate(k):                                                # same plausibility rule as pose_arm.arm_vectors, in cell units (1 cell = 13.3 px at 640)
    ls, rs, re, rw = k[0], k[1], k[2], k[3]; W = abs(rs[0] - ls[0])
    if min(ls[2], rs[2], re[2], rw[2]) < 0.07 or W < 1.5 or abs(rs[1] - ls[1]) > 0.6 * W: return False
    lu, lf = np.hypot(*(re[:2] - rs[:2])), np.hypot(*(rw[:2] - re[:2]))
    return 0.4 * W < lu < 2.6 * W and 0.3 * W < lf < 2.6 * W
def jitter(k, ok):                                          # mean frame-to-frame wrist+elbow motion (cells) between consecutive plausible frames
    d = [np.hypot(*(k[i + 1, j, :2] - k[i, j, :2])) for i in range(len(k) - 1) if ok[i] and ok[i + 1] for j in (2, 3)]
    return float(np.mean(d)) if d else None
ok0 = np.array([gate(k) for k in K0]); ok1 = np.array([gate(k) for k in K1]); both = ok0 & ok1
err = np.hypot(*(K1[..., :2] - KP[vi][..., :2]).transpose(2, 0, 1))[both] if both.any() else np.zeros((0, 4))
res = {"val_frames": int(len(vi)), "coverage_pct": {"original": round(100 * float(ok0.mean()), 1), "finetuned": round(100 * float(ok1.mean()), 1)},
       "jitter_cells_per_frame": {"original": jitter(K0, ok0), "finetuned": jitter(K1, ok1)},
       "finetuned_vs_teacher_labels_px_at_640": {n: round(float(err[:, k].mean() * 13.33), 1) for k, n in enumerate(["L_shoulder", "R_shoulder", "R_elbow", "R_wrist"])} if len(err) else None,
       "original_vs_teacher_labels_px_at_640": None}
e0 = np.hypot(*(K0[..., :2] - KP[vi][..., :2]).transpose(2, 0, 1))[both] if both.any() else None
if e0 is not None: res["original_vs_teacher_labels_px_at_640"] = {n: round(float(e0[:, k].mean() * 13.33), 1) for k, n in enumerate(["L_shoulder", "R_shoulder", "R_elbow", "R_wrist"])}
print(json.dumps(res, indent=1)); Path(str(a.out) + ".eval.json").write_text(json.dumps(res, indent=1))
