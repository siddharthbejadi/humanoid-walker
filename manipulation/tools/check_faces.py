"""QA for a video that shows a person: run a frontal-face detector over sampled frames and count hits (should be 0 after pixelation).
    python tools/check_faces.py VIDEO.mp4 [--every 5]      (needs an OpenCV with CascadeClassifier, e.g. opencv-python 4.x)
A detector can miss faces, so this is a check, not proof: also look at the video."""
import sys, cv2
every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 5
dump = sys.argv[sys.argv.index("--dump") + 1] if "--dump" in sys.argv else None      # save the flagged panes (with the detection boxed) to PREFIX_<frame>.png
F = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"); cap = cv2.VideoCapture(sys.argv[1]); n = hits = checked = 0
while True:
    ok, fr = cap.read()
    if not ok: break
    if n % every == 0:
        pane = fr[40:520, :640].copy(); checked += 1; found = F.detectMultiScale(cv2.cvtColor(pane, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(40, 40)); hits += len(found)
        if dump and len(found):
            for (x, y, w_, h_) in found: cv2.rectangle(pane, (x, y), (x + w_, y + h_), (0, 255, 255), 2)
            cv2.imwrite(f"{dump}_{n}.png", pane)
    n += 1
print(f"{sys.argv[1]}: {n} frames, {checked} checked, face-detector hits: {hits}")
