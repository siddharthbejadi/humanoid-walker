#!/usr/bin/env python3
"""Measure how well a hand-tracking model finds the user's hand in extracted frames.

    ~/hc_env/bin/python -I detector_check.py FRAMES_ROOT OUT_DIR REVIEW_DIR

FRAMES_ROOT/<clip>/*.jpg (15 fps).  For every clip:
  * runs MediaPipe Hands twice: per-frame ("static") and with tracking ("video"),
  * reports detection rate, confidence, how often the whole hand is inside the frame, hand size,
    position coverage (3x3 grid), frame-to-frame jumps (tracking failures), left/right label counts,
  * saves the 21 landmarks per frame to OUT_DIR/landmarks_<clip>.npz (derived numbers, no faces),
  * saves an annotated contact sheet to REVIEW_DIR (PRIVATE: faces may be visible, do not publish).
"""
import sys
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

root, out_dir, review_dir = (Path(p) for p in sys.argv[1:4])
out_dir.mkdir(parents=True, exist_ok=True)
review_dir.mkdir(parents=True, exist_ok=True)
mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils


def run(frames, static):
    hands = mp_hands.Hands(static_image_mode=static, max_num_hands=2, model_complexity=1,
                           min_detection_confidence=0.4, min_tracking_confidence=0.4)
    lms = np.full((len(frames), 21, 3), np.nan, np.float32)
    score = np.zeros(len(frames), np.float32)
    label = [""] * len(frames)
    nhands = np.zeros(len(frames), np.int8)
    for i, path in enumerate(frames):
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            nhands[i] = len(res.multi_hand_landmarks)
            best = int(np.argmax([h.classification[0].score for h in res.multi_handedness]))   # strongest hand
            lms[i] = [[p.x, p.y, p.z] for p in res.multi_hand_landmarks[best].landmark]
            score[i] = res.multi_handedness[best].classification[0].score
            label[i] = res.multi_handedness[best].classification[0].label
    hands.close()
    return lms, score, label, nhands


def summarize(lms, score, label, nhands):
    found = ~np.isnan(lms[:, 0, 0])
    n = len(found)
    inside = np.zeros(n, bool)
    inside[found] = np.all((lms[found, :, :2] > 0.0) & (lms[found, :, :2] < 1.0), axis=(1, 2))
    wrist = lms[:, 0, :2]
    size = np.full(n, np.nan)
    size[found] = np.linalg.norm(lms[found, :, :2].max(1) - lms[found, :, :2].min(1), axis=1)
    jump = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
    grid = np.zeros((3, 3), int)
    for p in wrist[found]:
        grid[min(2, int(np.clip(p[1], 0, 0.999) * 3)), min(2, int(np.clip(p[0], 0, 0.999) * 3))] += 1
    runs, cur = [], 0                                   # longest stretch without a detection
    for f in found:
        cur = 0 if f else cur + 1
        runs.append(cur)
    return dict(frames=n, detected=round(float(found.mean()), 3), whole_hand_inside=round(float(inside.mean()), 3),
                mean_score=round(float(score[found].mean()), 3) if found.any() else None,
                two_hands_frac=round(float((nhands >= 2).mean()), 3),
                labels={k: int(sum(1 for l in label if l == k)) for k in ("Left", "Right")},
                hand_size_median=round(float(np.nanmedian(size)), 3) if found.any() else None,
                big_jumps_over_0p25=int(np.nansum(jump > 0.25)), median_jump=round(float(np.nanmedian(jump)), 4),
                longest_gap_frames=int(max(runs)), grid_3x3=grid.tolist())


results = {}
for clip_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    frames = sorted(glob.glob(str(clip_dir / "*.jpg")))
    print(f"== {clip_dir.name}: {len(frames)} frames", flush=True)
    s_lms, s_score, s_label, s_n = run(frames, static=True)
    v_lms, v_score, v_label, v_n = run(frames, static=False)
    results[clip_dir.name] = dict(static=summarize(s_lms, s_score, s_label, s_n), video=summarize(v_lms, v_score, v_label, v_n))
    np.savez_compressed(out_dir / f"landmarks_{clip_dir.name}.npz", landmarks_video=v_lms, score_video=v_score,
                        landmarks_static=s_lms, score_static=s_score, fps=15)
    # annotated review sheet (private)
    pick = np.linspace(0, len(frames) - 1, 15).astype(int)
    tiles = []
    for i in pick:
        img = cv2.imread(frames[i])
        if not np.isnan(v_lms[i, 0, 0]):
            h, w = img.shape[:2]
            pts = [(int(p[0] * w), int(p[1] * h)) for p in v_lms[i]]
            for a, b in mp_hands.HAND_CONNECTIONS:
                cv2.line(img, pts[a], pts[b], (0, 255, 0), 3)
            for p in pts:
                cv2.circle(img, p, 5, (0, 0, 255), -1)
        else:
            cv2.putText(img, "NO HAND", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
        scale = 300 / img.shape[1] if img.shape[1] >= img.shape[0] else 240 / img.shape[1]
        tiles.append(cv2.resize(img, None, fx=scale, fy=scale))
    hh, ww = max(t.shape[0] for t in tiles), max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, ww - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
    rows = [np.hstack(tiles[r * 5:(r + 1) * 5]) for r in range(3)]
    cv2.imwrite(str(review_dir / f"overlay_{clip_dir.name}.png"), np.vstack(rows))

(out_dir / "detector_check.json").write_text(json.dumps(results, indent=1))
print(json.dumps({k: {m: {kk: vv for kk, vv in v.items() if kk in ("detected", "whole_hand_inside", "mean_score", "big_jumps_over_0p25", "longest_gap_frames")}
                      for m, v in d.items()} for k, d in results.items()}, indent=1))
