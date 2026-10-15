"""Side-by-side demo video: camera frames (head pixelated) | simulated humanoid copying the right arm. Same pipeline as demo_live.py.

    python tools/make_demo_video.py --out raw.mp4 --models-dir models_zoo --model st_movenet_192_int8.tflite \
        --segment FRAMES_DIR|FIRST|LAST|SRC_FPS|LABEL  [--segment ...]

FRAMES_DIR holds numbered .jpg frames (as saved by demo_live.py key 'c'); FIRST/LAST are frame indices; SRC_FPS is the real capture rate of those frames.
Frames show a person, so the head is ALWAYS pixelated (two independent estimates: MoveNet face keypoints and the shoulder line); the script then runs an
OpenCV face detector over the result and reports how many frames still show a face.  Output is mp4v; re-encode with ffmpeg for the web (see README)."""
import argparse, glob, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose_arm
from pose_arm import MoveNet, arm_vectors, forward_fractions, RS, RE, RW
from humanoid_view import Humanoid, ArmIK3

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--segment", action="append", required=True, help="DIR|FIRST|LAST|SRC_FPS|LABEL")
ap.add_argument("--model", default="st_movenet_192_int8.tflite"); ap.add_argument("--models-dir", default=None); ap.add_argument("--smooth", type=float, default=0.7)
ap.add_argument("--fps", type=float, default=15.0)
a = ap.parse_args()
if a.models_dir: pose_arm.MODELS = Path(a.models_dir)
net = MoveNet(192, a.model); robot = Humanoid(480, 480)
FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") if hasattr(cv2, "CascadeClassifier") else None   # OpenCV 5 dropped it: use tools/check_faces.py


class OneEuro:
    def __init__(s, mc=0.7, beta=0.03, dc=1.0): s.mc, s.beta, s.dc, s.x, s.dx, s.t = mc, beta, dc, None, 0.0, None
    @staticmethod
    def _a(c, dt): return 1.0 / (1.0 + 1.0 / (2 * np.pi * c * dt))
    def __call__(s, x, t):
        if s.x is None: s.x, s.t = x, t; return x
        dt = max(t - s.t, 1e-3); s.t = t; s.dx += s._a(s.dc, dt) * ((x - s.x) / dt - s.dx); s.x += s._a(s.mc + s.beta * abs(s.dx), dt) * (x - s.x); return s.x


def blur_head(img, kp):
    h, w = img.shape[:2]; mask = np.zeros((h, w), np.uint8); W = abs(kp[RS][0] - kp[5][0])
    if W > 20: cv2.circle(mask, (int((kp[RS][0] + kp[5][0]) / 2), int((kp[RS][1] + kp[5][1]) / 2 - 0.8 * W)), int(0.8 * W), 255, -1)       # from the shoulder line
    face = [kp[i] for i in range(5) if kp[i][2] > 0.03]
    if face: cv2.circle(mask, (int(np.mean([p[0] for p in face])), int(np.mean([p[1] for p in face]))), int(max(0.75 * W, 55)), 255, -1)     # from the face keypoints
    if not mask.any(): cv2.circle(mask, (int(kp[0][0]), int(kp[0][1])), 90, 255, -1)
    pix = cv2.resize(cv2.resize(img, (w // 22, h // 22), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_NEAREST)
    m = cv2.GaussianBlur(mask, (0, 0), 6)[..., None] / 255.0
    return (img * (1 - m) + pix * m).astype(np.uint8)


def txt(im, s, xy, scale=0.55, col=(255, 255, 255), th=1): cv2.putText(im, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


H, HEAD, FOOT = 480, 40, 54
writer = None; total = 0; faces_left = 0; checked = 0; found_n = 0
for seg in a.segment:
    d, first, last, src_fps, label = seg.split("|", 4); first, last, src_fps = int(first), int(last), float(src_fps)
    files = sorted(glob.glob(str(Path(d) / "*.jpg"))); n_out = int((last - first) * a.fps / src_fps)
    ik = ArmIK3(robot); filt = [OneEuro(a.smooth) for _ in range(3)]; q = ik.REST.copy(); miss = 0
    for k in range(n_out):
        img = cv2.imread(files[min(first + int(k * src_fps / a.fps), len(files) - 1)]); t = total / a.fps
        kp = net(img); v = arm_vectors(kp, size=(img.shape[1], img.shape[0])); found = v is not None
        if found: qn, _ = ik.fit(*v, *forward_fractions(kp)); q = np.array([f(x, t) for f, x in zip(filt, qn)]); miss = 0; found_n += 1
        else:
            miss += 1
            if miss > 15: q = 0.9 * q + 0.1 * ik.REST
        cam = blur_head(img, kp)
        if FACE is not None and checked % 5 == 0:                                                                  # QA: a face detector must not find a face in what we publish
            g = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY); faces_left += len(FACE.detectMultiScale(g, 1.1, 5, minSize=(40, 40)))
        checked += 1
        if found:
            p = kp[:, :2].astype(int)
            for i, j in ((RS, RE), (RE, RW)): cv2.line(cam, tuple(p[i]), tuple(p[j]), (0, 220, 0), 4, cv2.LINE_AA)
            for j in (RS, RE, RW): cv2.circle(cam, tuple(p[j]), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cam = cv2.resize(cam, (int(H * img.shape[1] / img.shape[0]), H))
        sim = cv2.cvtColor(robot.render([*q, 0.0]), cv2.COLOR_RGB2BGR)
        W_ = cam.shape[1] + sim.shape[1]; frame = np.full((HEAD + H + FOOT, W_, 3), 28, np.uint8)
        frame[HEAD:HEAD + H, :cam.shape[1]] = cam; frame[HEAD:HEAD + H, cam.shape[1]:] = sim
        txt(frame, "Board camera (head pixelated)", (10, 26), 0.6, (240, 240, 240)); txt(frame, "Simulated humanoid: right arm", (cam.shape[1] + 10, 26), 0.6, (240, 240, 240))
        cv2.rectangle(frame, (0, HEAD), (10 + 9 * len(label), HEAD + 34), (28, 28, 28), -1); txt(frame, label, (10, HEAD + 24), 0.55, (255, 255, 255), 1)
        txt(frame, "arm found" if found else "arm lost - easing to rest", (cam.shape[1] + 10, HEAD + 24), 0.55, (120, 230, 120) if found else (160, 160, 160))
        txt(frame, "MoveNet int8 -> plausibility gate -> reachability-aware IK + depth cue -> One-Euro filter -> MuJoCo", (10, HEAD + H + 20), 0.5, (225, 225, 225))
        txt(frame, "Simulation only: the real arm is not driven here. Only the person's right arm (image left) is copied; no gripper.", (10, HEAD + H + 42), 0.5, (170, 190, 255))
        if writer is None: writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame); total += 1
writer.release()
print(f"wrote {a.out}: {total} frames ({total / a.fps:.1f} s), arm found in {100 * found_n / total:.1f}%; face detector hits on published frames: " + (f"{faces_left} in {-(-checked // 5)} checked" if FACE is not None else "not run here (OpenCV 5): run tools/check_faces.py"))
