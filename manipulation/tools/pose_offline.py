"""MoveNet -> IK -> humanoid on a folder of frames. Writes an mp4 (camera | skeleton | humanoid) and prints match statistics.
Contains camera pixels: keep PRIVATE.   usage: pose_offline.py FRAMES_DIR OUT.mp4 [--no-video]"""
import sys, glob, time, json, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_arm import MoveNet, arm_vectors, RS, RE, RW
from humanoid_view import Humanoid, ArmIK

folder, out = sys.argv[1], sys.argv[2]
net, robot = MoveNet(192), Humanoid(360, 360); ik = ArmIK(robot)
files = sorted(glob.glob(folder + "/*.jpg")); w = None
kps = None; q = ik.REST.copy(); lost = 0; res_u, res_f, found, clip = [], [], [], []
for f in files:
    img = cv2.imread(f); kp = net(img)
    if kps is not None: kp[:, :2] = 0.5 * kp[:, :2] + 0.5 * kps[:, :2] if kp[[RS, RE, RW], 2].min() > 0.07 else kps[:, :2]
    kps = kp
    v = arm_vectors(kp, size=(img.shape[1], img.shape[0]))
    if v is not None:
        qn, (eu, ef) = ik.fit(*v); q = 0.5 * q + 0.5 * qn; lost = 0; found.append(True); res_u.append(eu); res_f.append(ef); clip.append(ik.clipped)
    else:
        lost += 1; found.append(False)
        if lost > 15: q = 0.9 * q + 0.1 * ik.REST
    if out.endswith(".mp4") and "--no-video" not in sys.argv:
        prev = img.copy()
        if v is not None:
            for i, j in ((RS, RE), (RE, RW)): cv2.line(prev, tuple(kp[i, :2].astype(int)), tuple(kp[j, :2].astype(int)), (0, 255, 0), 4)
            for j in (RS, RE, RW): cv2.circle(prev, tuple(kp[j, :2].astype(int)), 8, (0, 0, 255), -1)
        prev = cv2.resize(prev, (int(360 * img.shape[1] / img.shape[0]), 360))
        tile = np.hstack([prev, cv2.cvtColor(robot.render(np.array([q[0], q[1], q[2], 0.0])), cv2.COLOR_RGB2BGR)])
        if w is None: w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (tile.shape[1], 360))
        w.write(tile)
if w is not None: w.release()
ru, rf = np.array(res_u), np.array(res_f)
s = dict(frames=len(files), arm_found_pct=round(100 * np.mean(found), 1),
         upper_arm_match_err_deg=dict(median=round(float(np.median(ru)), 1), p90=round(float(np.percentile(ru, 90)), 1)),
         forearm_match_err_deg=dict(median=round(float(np.median(rf)), 1), p90=round(float(np.percentile(rf, 90)), 1)),
         frames_beyond_robot_range_pct=round(100 * float(np.mean(clip)), 1), frames_within_15deg_both_pct=round(100 * float(np.mean((ru < 15) & (rf < 15))), 1))
print(json.dumps(s))
