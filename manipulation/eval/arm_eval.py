"""Student vs teacher in robot space.  arm_eval.py PRED.npz OUT_PREFIX [--video]
Metric: angle between student and teacher arm directions (upper arm, forearm) in the robot's front view, via the same kinematics as the IK.
Baseline: the mean teacher pose (what you get if the network learned nothing)."""
import sys, json, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armcopy.sim import Humanoid, ArmIK
d = np.load(sys.argv[1]); out = sys.argv[2]
P, Q, V = d["pred"], d["q"], d["valid"]
h = Humanoid(300, 300); ik = ArmIK(h)
ang = lambda a, b: np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))
D = lambda q: np.array([ik._dirs(x) for x in q])
dt, ds = D(Q[V]), D(P[V, :3]); db = D(np.tile(Q[V].mean(0), (V.sum(), 1)))
res = {"n_arm_frames": int(V.sum()),
       "student_err_deg": {"upper_median": round(float(np.median(ang(dt[:, :2], ds[:, :2]))), 1), "upper_p90": round(float(np.percentile(ang(dt[:, :2], ds[:, :2]), 90)), 1),
                           "fore_median": round(float(np.median(ang(dt[:, 2:], ds[:, 2:]))), 1), "fore_p90": round(float(np.percentile(ang(dt[:, 2:], ds[:, 2:]), 90)), 1)},
       "mean_pose_baseline_err_deg": {"upper_median": round(float(np.median(ang(dt[:, :2], db[:, :2]))), 1), "fore_median": round(float(np.median(ang(dt[:, 2:], db[:, 2:]))), 1)},
       "within_15deg_pct": round(100 * float(np.mean((ang(dt[:, :2], ds[:, :2]) < 15) & (ang(dt[:, 2:], ds[:, 2:]) < 15))), 1),
       "present_acc": round(float(((P[:, 3] > 0) == V).mean()), 3)}
print(json.dumps(res)); Path(out + "_arm_eval.json").write_text(json.dumps(res, indent=1))
if "--video" in sys.argv:
    w = cv2.VideoWriter(out + "_student_vs_teacher.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (600, 300))
    for p, q, v in list(zip(P, Q, V))[:600]:
        a = cv2.cvtColor(h.render([*p[:3], 0]), cv2.COLOR_RGB2BGR); b = cv2.cvtColor(h.render([*q, 0]), cv2.COLOR_RGB2BGR)
        cv2.putText(a, "student" + ("" if p[3] > 0 else " (no arm)"), (6, 20), 0, 0.55, (0, 255, 255), 2); cv2.putText(b, "teacher" + ("" if v else " (no arm)"), (6, 20), 0, 0.55, (0, 255, 0), 2)
        w.write(np.hstack([a, b]))
    w.release()
