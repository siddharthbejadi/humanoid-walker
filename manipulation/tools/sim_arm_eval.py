#!/usr/bin/env python3
"""Drive the simulated arm from the student's predictions and from the teacher's targets; report fingertip error.

    ~/hc_train/bin/python -I sim_arm_eval.py PRED.npz OUT_DIR TAG --cache CACHE.npz [--holdout-clip NAME] [--video]

PRED.npz comes from train_student.py (pred, target, clip, frame). Teacher = hand-tracking targets. Baseline = an arm that
always holds the average training pose (shows that the student learned something beyond the mean).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hc_common import retarget
from arm_model import build_model, set_pose
import mujoco

ap = argparse.ArgumentParser()
ap.add_argument("pred", type=Path); ap.add_argument("out", type=Path); ap.add_argument("tag")
ap.add_argument("--cache", type=Path, required=True); ap.add_argument("--holdout-clip", default=None)
ap.add_argument("--video", action="store_true")
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

p = np.load(args.pred); P, T, clip, frame = p["pred"], p["target"], p["clip"], p["frame"]
cache = np.load(args.cache); Y, cid, names = cache["targets"], cache["clip_id"], list(cache["clip_names"])
train_mask = cid != names.index(args.holdout_clip) if args.holdout_clip else np.ones(len(Y), bool)
mean_pose = retarget(Y[train_mask].mean(0, keepdims=True))[0]

model = build_model(); data = mujoco.MjData(model)
Jt, Js = retarget(T), retarget(np.clip(P, [0, 0, 0, 0], [1, 1, 1, 1]))
order = np.lexsort((frame, clip)); Jt, Js, clip, frame = Jt[order], Js[order], clip[order], frame[order]
ee_t = np.array([set_pose(model, data, "teacher_", j) for j in Jt])
ee_s = np.array([set_pose(model, data, "student_", j) for j in Js])
ee_m = np.array([set_pose(model, data, "student_", mean_pose) for _ in Jt])
err_s = np.linalg.norm(ee_s - ee_t, axis=1) * 100                 # cm
err_m = np.linalg.norm(ee_m - ee_t, axis=1) * 100
jerr = np.degrees(np.sqrt(((Js - Jt) ** 2).mean(0)))
travel = float(np.linalg.norm(np.diff(ee_t, axis=0), axis=1).sum() * 100 / max(1, len(ee_t) - 1) * 15)   # cm/s of teacher fingertip
res = dict(tag=args.tag, frames=int(len(Jt)),
           fingertip_error_cm=dict(mean=round(float(err_s.mean()), 2), median=round(float(np.median(err_s)), 2), p90=round(float(np.percentile(err_s, 90)), 2)),
           baseline_mean_pose_error_cm=dict(mean=round(float(err_m.mean()), 2), median=round(float(np.median(err_m)), 2), p90=round(float(np.percentile(err_m, 90)), 2)),
           within_2cm=round(float((err_s < 2).mean()), 3), within_5cm=round(float((err_s < 5).mean()), 3),
           baseline_within_5cm=round(float((err_m < 5).mean()), 3),
           joint_rmse_deg=dict(zip(["shoulder", "arm", "elbow", "gripper"], np.round(jerr, 1).tolist())),
           gripper_open_closed_accuracy=round(float(((Js[:, 3] > 0.5) == (Jt[:, 3] > 0.5)).mean()), 3),
           teacher_fingertip_speed_cm_s=round(travel, 1))
(args.out / f"sim_eval_{args.tag}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
t = np.arange(len(Jt)) / 15.0
for k, lab in enumerate(["x (sideways, cm)", "y (forward, cm)", "z (up, cm)"]):
    ax[k].plot(t, 100 * ee_t[:, k], color="tab:green", label="teacher (hand tracking)"); ax[k].plot(t, 100 * ee_s[:, k], color="tab:blue", label="student")
    ax[k].set_ylabel(lab)
ax[3].plot(t, Jt[:, 3], color="tab:green"); ax[3].plot(t, Js[:, 3], color="tab:blue"); ax[3].set_ylabel("gripper (rad)"); ax[3].set_xlabel("time (s), labelled frames only")
ax[0].legend(loc="upper right"); fig.suptitle(f"Simulated arm fingertip: student vs teacher ({args.tag})"); fig.tight_layout()
fig.savefig(args.out / f"sim_eval_{args.tag}.png", dpi=110)

if args.video:
    import imageio.v2 as imageio
    r = mujoco.Renderer(model, 360, 360); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.08, -0.12]; cam.distance, cam.azimuth, cam.elevation = 0.95, 150, -12
    with imageio.get_writer(args.out / f"sim_eval_{args.tag}.mp4", fps=15) as w:
        for js, jt in zip(Js, Jt):
            set_pose(model, data, "teacher_", jt); set_pose(model, data, "student_", js)
            r.update_scene(data, cam); w.append_data(r.render())
    print("wrote video", args.out / f"sim_eval_{args.tag}.mp4")
