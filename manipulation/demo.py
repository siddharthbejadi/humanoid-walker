#!/usr/bin/env python
"""LIVE DEMO: board camera -> STUDENT (fine-tuned zoo MoveNet, int8) -> arm keypoints -> IK (+ depth cue) -> simulated humanoid right arm.

    python demo.py [--index 0] [--rotate 180]
Panes: camera (detected right arm drawn on it) | humanoid driven by the STUDENT.   m = add the ORIGINAL zoo MoveNet pane for comparison,  g = gripper on/off, e = real arm to HOME, k = real arm RELAX (torque off), u = calibrate with hand OPEN, f = calibrate with a FIST (hold the pose ~1.5 s),
o = add the old image->joints network (experiment).   c = save RAW frames (private, red dot)   r = record the robot panes only (no camera pixels)   q = quit."""
import argparse, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from armcopy.sim import Humanoid, ArmIK3
from armcopy.pose import MoveNet, arm_vectors, forward_fractions, RS, RE, RW
from armcopy.grip import hand_crop
from armcopy.filters import OneEuro
import onnxruntime as ort
ort.set_default_logger_severity(3)

root = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--index", type=int, default=0); ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270]); ap.add_argument("--url", default=None)
ap.add_argument("--grip-lo", type=float, default=0.15, help="raw grip value mapped to fully open (the network is conservative: open hands read ~0.1)")
ap.add_argument("--grip-hi", type=float, default=0.6, help="raw grip value mapped to fully closed")
ap.add_argument("--smooth", type=float, default=0.7, help="arm smoothing cut-off in Hz: lower = smoother but laggier (try 0.4 .. 1.5)")
ap.add_argument("--robot-port", default=None, help="COM port of the STM32 (e.g. COM3): drive the REAL right arm too")
ap.add_argument("--live", action="store_true", help="really send servo frames (without it the robot output is a dry run)")
ap.add_argument("--robot-speed", type=float, default=200.0, help="max servo speed in us/s (slow = safe)")
ap.add_argument("--model", default=None, help="keypoint model in models_zoo/ (default: the fine-tuned int8 model if present, else the original ST zoo model from models/fetch_zoo_models.py)")
ap.add_argument("--save-dir", type=Path, default=Path(__file__).resolve().parent / "private_captures")
a = ap.parse_args()
ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

robot = Humanoid(360, 360)
class Chain:                                                   # one keypoint model + its own IK state and smoothing
    def __init__(self, name, label, colour):
        self.net, self.ik, self.label, self.colour = MoveNet(192, name), ArmIK3(robot), label, colour
        self.q, self.miss, self.ms, self.v, self.kp, self.grip = self.ik.REST.copy(), 0, 0.0, None, None, 0.0
        self.filt = [OneEuro(a.smooth) for _ in range(3)]
    def step(self, frame):
        h, w = frame.shape[:2]; t0 = time.perf_counter(); self.kp = self.net(frame); self.ms = 0.9 * self.ms + 0.1 * (time.perf_counter() - t0) * 1000
        self.v = arm_vectors(self.kp, size=(w, h))
        if self.v is not None:
            q, _ = self.ik.fit(*self.v, *forward_fractions(self.kp)); now = time.time(); self.q = np.array([f(x, now) for f, x in zip(self.filt, q)]); self.miss = 0
        else:
            self.miss += 1
            if self.miss > 15: self.q = 0.9 * self.q + 0.1 * self.ik.REST            # arm lost ~1 s -> ease to rest
    def pane(self):
        im = cv2.cvtColor(robot.render([*self.q, self.grip * 1.39]), cv2.COLOR_RGB2BGR)
        cv2.putText(im, f"{self.label}  {self.ms:4.1f} ms  arm {'OK' if self.v is not None else 'lost'}", (8, 22), 0, 0.5, self.colour, 2)
        cv2.rectangle(im, (8, 340), (108, 354), (255, 255, 255), 1); cv2.rectangle(im, (8, 340), (8 + int(100 * self.grip), 354), (0, 0, 255), -1); cv2.putText(im, 'gripper', (114, 352), 0, 0.45, (255, 255, 255), 1)
        return im

GRIP_MODEL = next((m for m in (root / "weights" / "grip_v2.onnx", root / "export" / "grip_v2.onnx") if m.exists()), root / "weights" / "grip_v2.onnx")   # optional; models/train_grip.py makes it
if GRIP_MODEL.exists(): grip_sess = ort.InferenceSession(str(GRIP_MODEL), providers=["CPUExecutionProvider"]); grip_on = True
else: grip_sess = None; grip_on = False; print("gripper model not found (weights/grip_v2.onnx): gripper disabled, run models/train_grip.py to create it")
grip_raw_init = 0; grip_raw = 0.0; grip_med = 0.0; crop_view = None
import collections; raw_hist = collections.deque(maxlen=7); grip_closed = False; g_open, g_close = 0.2, 0.35; vals = {}; calib = None
from armcopy import pose as _pose
FT, ZOO = "st_movenet_ft_192_int8.tflite", "st_movenet_192_int8.tflite"
model_name = a.model or (FT if (_pose.MODELS / FT).exists() else ZOO)
if not (_pose.MODELS / model_name).exists(): sys.exit(f"missing {_pose.MODELS / model_name}: run  python models/fetch_zoo_models.py")
student = Chain(model_name, "STUDENT (fine-tuned MoveNet+IK)" if model_name == FT else "MoveNet (ST zoo) + IK", (0, 255, 255))
original = None; old_sess = None; old_q = None   # m: compares against the original zoo model
def open_cam():
    tried = []
    for idx in [a.index] + [i for i in range(6) if i != a.index]:
        for name, be in (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)):
            c = cv2.VideoCapture(idx, be); ok = c.isOpened() and c.read()[0]; tried.append(f"{idx}/{name}:{'OK' if ok else 'no'}")
            if ok: print("camera opened:", idx, name); return c
            c.release()
    print("tried", " ".join(tried)); return None
cap = cv2.VideoCapture(a.url) if a.url else open_cam()
if cap is None or not cap.isOpened():
    sys.exit("No camera could be opened. Close every other window using it (another demo/record_uvc/Teams), unplug and replug the board's USB, then retry.")
real = None
if a.robot_port or a.live:
    from armcopy.robot import RightArm
    real = RightArm(a.robot_port, live=bool(a.live and a.robot_port), max_us_per_s=a.robot_speed)
    print("REAL ARM:", "LIVE on " + a.robot_port if real.live else "dry run (no frames sent)", "| limits:", {n: (v["min_us"], v["max_us"], "measured" if v["measured"] else "walking-default") for n, v in real.j.items()})
fps, t_prev, save_to, save_n, last_save, writer = 0.0, time.time(), None, 0, 0.0, None
while True:
    ok, frame = cap.read()
    if not ok: break
    if a.rotate: frame = cv2.rotate(frame, ROT[a.rotate])
    h, w = frame.shape[:2]
    student.step(frame)
    if grip_on and student.v is not None:                       # gripper: hand crop beyond the wrist -> FreiHAND-trained network (no fine-tuning)
        crop, _ = hand_crop(frame, student.kp)
        grip_raw = float(1 / (1 + np.exp(-grip_sess.run(None, {"image": crop[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None]})[0][0])))
        raw_hist.append(grip_raw); grip_med = float(np.median(raw_hist)); crop_view = crop
        if calib is not None:                                     # calibration: collect raw values for ~1.5 s while you hold the pose
            calib[1].append(grip_raw)
            if time.time() - calib[2] > 1.5:
                vals[calib[0]] = float(np.median(calib[1])); calib = None
                if "open" in vals and "closed" in vals and vals["closed"] - vals["open"] > 0.05:
                    mid = 0.5 * (vals["open"] + vals["closed"]); half = 0.25 * (vals["closed"] - vals["open"]); g_open, g_close = mid - half, mid + half
        if grip_closed and grip_med < g_open: grip_closed = False
        elif not grip_closed and grip_med > g_close: grip_closed = True
        student.grip += 0.3 * ((1.0 if grip_closed else 0.0) - student.grip)
    elif student.v is None:
        student.grip *= 0.9; crop_view = None
    if not grip_on: student.grip = 0.0
    if real is not None and student.v is not None: real.set_angles(student.q[0], student.q[1], student.q[2], student.grip * 1.39)
    elif real is not None and student.miss > 15: real.go_home()
    cam = frame.copy()
    if student.v is not None:                                   # draw the detected right arm on the camera view
        p = student.kp[:, :2].astype(int)
        for i, j in ((RS, RE), (RE, RW)): cv2.line(cam, tuple(p[i]), tuple(p[j]), (0, 255, 0), 4)
        for j in (RS, RE, RW): cv2.circle(cam, tuple(p[j]), 8, (0, 0, 255), -1)
    cam = cv2.resize(cam, (int(360 * w / h), 360))
    if crop_view is not None and grip_on:                        # inset: what the gripper network sees + its raw output
        cam[8:104, cam.shape[1] - 104:cam.shape[1] - 8] = crop_view; cv2.putText(cam, f"raw {grip_med:.2f}", (cam.shape[1] - 104, 122), 0, 0.5, (0, 255, 255), 2)
        lab = ("HAND: CLOSED", (0, 0, 255)) if grip_closed else ("HAND: OPEN", (0, 200, 0)); cv2.putText(cam, lab[0], (8, 346), 0, 0.9, lab[1], 3)
        cv2.putText(cam, f"open<{g_open:.2f} closed>{g_close:.2f}" + (" CALIBRATING..." if calib else ""), (8, 322), 0, 0.5, (255, 255, 255), 1)
    sp = student.pane()
    if real is not None: cv2.putText(sp, ("REAL ARM LIVE" if real.live else "real arm: dry run") + f"  sh {real.cur['shoulder']:.0f} arm {real.cur['arm']:.0f} el {real.cur['elbow']:.0f} gr {real.cur['grip']:.0f}", (8, 44), 0, 0.42, (0, 0, 255) if real.live else (200, 200, 200), 1)
    panes = [cam, sp]
    if original is not None: original.step(frame); panes.append(original.pane())
    if old_sess is not None:
        sq = np.zeros((max(h, w),) * 2 + (3,), np.uint8); s = max(h, w); sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = frame
        x = cv2.resize(sq, (192, 192), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None]
        o = old_sess.run(None, {"image": x})[0][0]; old_q = (0.6 * old_q + 0.4 * (o[:3] if o[3] > 0 else student.ik.REST))
        im = cv2.cvtColor(robot.render([*old_q, 0]), cv2.COLOR_RGB2BGR); cv2.putText(im, "OLD image->joints net (experiment)", (8, 22), 0, 0.5, (255, 128, 0), 2); panes.append(im)
    if save_to is not None and time.time() - last_save >= 1 / 15:
        cv2.imwrite(str(save_to / f"{save_n:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]); save_n += 1; last_save = time.time()
        cv2.circle(panes[0], (panes[0].shape[1] - 20, 20), 9, (0, 0, 255), -1)
    now = time.time(); fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6); t_prev = now
    cv2.putText(panes[0], f"fps {fps:4.1f}", (8, 346), 0, 0.6, (0, 255, 255), 2)
    cv2.imshow("camera | student | (m: original, o: old net)", np.hstack(panes))
    if writer is not None: writer.write(np.hstack(panes[1:]))
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"): break
    if k == ord("g") and grip_sess is not None: grip_on = not grip_on
    if k == ord("e") and real is not None: real.go_home()
    if k == ord("k") and real is not None: real.relax()
    if k in (ord("u"), ord("f")) and student.v is not None: calib = ("open" if k == ord("u") else "closed", [], time.time())   # hold the pose, then press
    if k == ord("m"): original = None if original else Chain("st_movenet_192_int8.tflite", "ORIGINAL zoo MoveNet+IK", (0, 255, 0))
    if k == ord("o"):
        if old_sess: old_sess = None
        else:
            import onnxruntime as ort; ort.set_default_logger_severity(3)
            old_sess = ort.InferenceSession(str(root / "export" / "arm_final_v1_float.onnx"), providers=["CPUExecutionProvider"]); old_q = student.ik.REST.copy()
    if k == ord("c"):
        if save_to is None: save_to = a.save_dir / time.strftime("live_%H%M%S"); save_to.mkdir(parents=True, exist_ok=True); save_n = 0
        else: save_to = None
    if k == ord("r"):
        if writer is None: writer = cv2.VideoWriter(str(root / "results" / "live_demo_robot_only.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (360 * (len(panes) - 1), 360))
        else: writer.release(); writer = None
if writer is not None: writer.release()
if real is not None: real.close()
cap.release(); cv2.destroyAllWindows()
