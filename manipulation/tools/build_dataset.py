#!/usr/bin/env python3
"""Build the training set from extracted frames + hand landmarks.

    ~/hc_train/bin/python -I build_dataset.py FRAME_ROOT_1 [FRAME_ROOT_2 ...] --landmarks DIR --out DERIVED_DIR --cache CACHE.npz

Per clip: targets from landmarks (whole hand inside the frame), short gaps (<=5 frames) interpolated, light smoothing.
Writes DERIVED_DIR/targets_<clip>.npz (numbers only, safe to publish) and a PRIVATE image cache (128x128, never commit).
"""
import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hc_common import fill_short_gaps, smooth, targets_from_landmarks, NAMES

ap = argparse.ArgumentParser()
ap.add_argument("frame_roots", nargs="+", type=Path)
ap.add_argument("--landmarks", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--cache", type=Path, required=True)
ap.add_argument("--exclude", nargs="*", default=["IMG_0029"], help="clips left out (IMG_0029: hand too close, both hands in view)")
ap.add_argument("--size", type=int, default=128)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

images, targets, clip_id, frame_id, flags, clip_names = [], [], [], [], [], []
summary = {}
for root in args.frame_roots:
    for clip_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = clip_dir.name
        if name in args.exclude:
            print(f"skip {name} (excluded)")
            continue
        frames = sorted(glob.glob(str(clip_dir / "*.jpg")))
        data = np.load(args.landmarks / f"landmarks_{name}.npz")
        lm = data["landmarks_video"]
        assert len(lm) == len(frames), (name, len(lm), len(frames))
        h, w = cv2.imread(frames[0]).shape[:2]
        found = ~np.isnan(lm[:, 0, 0])
        inside = np.zeros(len(lm), bool)
        inside[found] = np.all((lm[found, :, :2] > 0.0) & (lm[found, :, :2] < 1.0), axis=(1, 2))
        tg = np.zeros((len(lm), 4), np.float32)
        for i in np.flatnonzero(inside):
            tg[i] = targets_from_landmarks(lm[i], w, h)
        tg, valid, interp = fill_short_gaps(tg, inside)
        tg = smooth(tg, valid)
        cid = len(clip_names)
        clip_names.append(name)
        keep = np.flatnonzero(valid)
        for i in keep:
            img = cv2.resize(cv2.imread(frames[i]), (args.size, args.size), interpolation=cv2.INTER_AREA)   # stretch to square
            images.append(img[:, :, ::-1])                                                                    # BGR -> RGB
            targets.append(tg[i]); clip_id.append(cid); frame_id.append(i); flags.append(interp[i])
        np.savez_compressed(args.out / f"targets_{name}.npz", targets=tg, valid=valid, interpolated=interp, fps=15, names=NAMES,
                            frame_width=w, frame_height=h)
        summary[name] = dict(frames=len(frames), labelled=int(valid.sum()), interpolated=int(interp.sum()), aspect=f"{w}x{h}")
        print(f"{name}: {len(frames)} frames -> {int(valid.sum())} labelled ({int(interp.sum())} interpolated)")

np.savez_compressed(args.cache, images=np.stack(images).astype(np.uint8), targets=np.stack(targets), clip_id=np.array(clip_id),
                    frame_id=np.array(frame_id), interpolated=np.array(flags), clip_names=np.array(clip_names))
t = np.stack(targets)
summary["total_labelled"] = int(len(t))
summary["target_ranges"] = {n: [round(float(t[:, k].min()), 3), round(float(np.percentile(t[:, k], 50)), 3), round(float(t[:, k].max()), 3)]
                            for k, n in enumerate(NAMES)}
(args.out / "dataset_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
