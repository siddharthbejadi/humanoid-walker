"""camera frame | humanoid driven by the STUDENT | humanoid driven by the TEACHER (MoveNet+IK), for the frames in a student prediction file.
    student_sim_video.py PRED.npz FRAMES_DIR OUT.mp4      (contains camera pixels: keep PRIVATE)"""
import sys, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armcopy.sim import Humanoid
d = np.load(sys.argv[1]); frames = Path(sys.argv[2]); P, Q, V, F = d["pred"], d["q"], d["valid"], d["frame"]
h = Humanoid(360, 360); w = None
for p, q, v, f in zip(P, Q, V, F):
    cam = cv2.imread(str(frames / f"{int(f) + 1:05d}.jpg")); cam = cv2.resize(cam, (480, 360))
    a = cv2.cvtColor(h.render([*p[:3], 0] if p[3] > 0 else [0, -0.38, 0, 0]), cv2.COLOR_RGB2BGR)       # student says "no arm" -> rest pose
    b = cv2.cvtColor(h.render([*q, 0] if v else [0, -0.38, 0, 0]), cv2.COLOR_RGB2BGR)
    cv2.putText(a, "STUDENT" + ("" if p[3] > 0 else " (no arm)"), (8, 22), 0, 0.6, (0, 255, 255), 2)
    cv2.putText(b, "TEACHER" + ("" if v else " (no arm)"), (8, 22), 0, 0.6, (0, 255, 0), 2)
    tile = np.hstack([cam, a, b])
    if w is None: w = cv2.VideoWriter(sys.argv[3], cv2.VideoWriter_fourcc(*"mp4v"), 15, (tile.shape[1], 360))
    w.write(tile)
w.release(); print("wrote", sys.argv[3], len(P), "frames")
