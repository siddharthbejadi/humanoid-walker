"""Render the humanoid following the student's predictions on a recorded clip (arm-only video, no camera pixels)."""
import sys, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from humanoid_view import Humanoid, humanoid_angles
pred = np.load(sys.argv[1]); out = sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 400
P, T = pred["pred"][:n], pred["target"][:n]
h = Humanoid(360, 360); w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (720, 360))
for p, t in zip(P, T):
    a = cv2.cvtColor(h.render(humanoid_angles(p)), cv2.COLOR_RGB2BGR); b = cv2.cvtColor(h.render(humanoid_angles(t)), cv2.COLOR_RGB2BGR)
    cv2.putText(a, "student", (8, 22), 0, 0.6, (0, 255, 255), 2); cv2.putText(b, "teacher (hand tracking)", (8, 22), 0, 0.6, (0, 255, 0), 2)
    w.write(np.hstack([a, b]))
w.release(); print("wrote", out)
