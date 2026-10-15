"""Record numbered JPG frames from a camera (the STM32N6 board's UVC camera, a laptop webcam, or a phone stream) at a fixed rate.

    python data_collection/record_frames.py OUT_DIR [--index 0] [--rotate 0] [--fps 15] [--seconds 60] [--url http://IP:8080/video]

q stops early. The frames show a person: keep them private (the repo ignores private_captures/). This is the whole 'data collection rig':
a camera plus a person moving their right arm in front of it; labels come from a pre-trained pose model (preprocessing/), not from hand annotation."""
import argparse, sys, time
from pathlib import Path
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path); ap.add_argument("--index", type=int, default=0); ap.add_argument("--url", default=None)
ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270]); ap.add_argument("--fps", type=float, default=15.0); ap.add_argument("--seconds", type=float, default=0.0)
a = ap.parse_args()
ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
cap = cv2.VideoCapture(a.url) if a.url else cv2.VideoCapture(a.index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
if not cap.isOpened(): sys.exit("camera not opened (close other apps using it; try --index N)")
a.out.mkdir(parents=True, exist_ok=True); n, t0, last = 0, time.time(), 0.0
while True:
    ok, frame = cap.read()
    if not ok: break
    if a.rotate: frame = cv2.rotate(frame, ROT[a.rotate])
    now = time.time()
    if now - last >= 1.0 / a.fps:
        cv2.imwrite(str(a.out / f"{n:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]); n += 1; last = now
    cv2.imshow("recording (q = stop)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q") or (a.seconds and now - t0 > a.seconds): break
print(f"saved {n} frames to {a.out}"); cap.release(); cv2.destroyAllWindows()
