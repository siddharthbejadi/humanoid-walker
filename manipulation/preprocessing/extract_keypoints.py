"""Frames -> pose keypoints (no pixels): the preprocessing step that turns a private recording into shareable data.

    python preprocessing/extract_keypoints.py FRAMES_DIR OUT.json --fps 10.72 [--model st_movenet_192_int8.tflite]

OUT.json holds MoveNet's 17 keypoints per frame as [x, y, score] in image pixels, the frame size and the capture rate. replay.py turns such a file into the
simulated humanoid's motion with no camera and no images."""
import argparse, glob, json, sys
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armcopy.pose import MoveNet

ap = argparse.ArgumentParser()
ap.add_argument("frames", type=Path); ap.add_argument("out", type=Path); ap.add_argument("--fps", type=float, required=True); ap.add_argument("--model", default="st_movenet_192_int8.tflite")
a = ap.parse_args()
net = MoveNet(192, a.model); files = sorted(glob.glob(str(a.frames / "*.jpg"))); kps = []
for f in files:
    img = cv2.imread(f); size = [img.shape[1], img.shape[0]]
    kps.append([[round(float(v), 2) if i < 2 else round(float(v), 4) for i, v in enumerate(p)] for p in net(img)])
a.out.parent.mkdir(parents=True, exist_ok=True)
json.dump({"_note": "MoveNet keypoints (COCO order, x y score in pixels); no images. Source: one live recording from the STM32N6 board camera.", "model": a.model, "fps": a.fps, "frame_size": size, "keypoints": kps}, open(a.out, "w"), separators=(",", ":"))
print(f"{len(kps)} frames -> {a.out} ({a.out.stat().st_size / 1024:.0f} KB)")
