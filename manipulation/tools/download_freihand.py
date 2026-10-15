"""Partial FreiHAND download (HTTP range requests; no 3.9 GB archive): keypoint files + a spread-out subset of training images.
    download_freihand.py OUT_DIR [N_IMAGES=6000]      (needs remotezip).  FreiHAND is for non-commercial research use (see its licence)."""
import sys, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from remotezip import RemoteZip
URL = "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip"
out = Path(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
(out / "rgb").mkdir(parents=True, exist_ok=True)
with RemoteZip(URL) as z:
    names = {i.filename: i for i in z.infolist()}
    print(len(names), "entries; sample:", list(names)[:3])
    for f in ("training_xyz.json", "training_K.json"):
        key = next(k for k in names if k.endswith(f)); (out / f).write_bytes(z.read(key))
    prefix = next(k for k in names if k.endswith("training/rgb/00000000.jpg")).rsplit("/", 1)[0]
total = 130240; ids = [int(i * total / n) for i in range(n)]
def get(chunk):
    with RemoteZip(URL) as z:
        for i in chunk:
            p = out / "rgb" / f"{i:08d}.jpg"
            if not p.exists(): p.write_bytes(z.read(f"{prefix}/{i:08d}.jpg"))
    return len(chunk)
chunks = [ids[k::12] for k in range(12)]
with ThreadPoolExecutor(12) as ex: done = sum(ex.map(get, chunks))
json.dump(ids, open(out / "ids.json", "w")); print("downloaded", done, "images to", out)
