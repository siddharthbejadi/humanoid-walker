"""Download the ST model-zoo MoveNet files this project starts from (not redistributed here: ST model zoo, SLA0044).
    python tools/fetch_zoo_models.py          -> manipulation/models_zoo/{st_movenet_192_int8.tflite, st_movenet_192.keras}"""
import urllib.request
from pathlib import Path
BASE = "STMicroelectronics/stm32ai-modelzoo/main/pose_estimation/movenet/ST_pretrainedmodel_public_dataset/coco_person_17kpts"
FILES = {  # (url, name): int8 tflite is a Git-LFS file (media host), the .keras file is a normal file (raw host)
    "st_movenet_192_int8.tflite": f"https://media.githubusercontent.com/media/{BASE}/st_movenet_lightning_a100_heatmaps_192/st_movenet_lightning_a100_heatmaps_192_int8.tflite",
    "st_movenet_192.keras": f"https://raw.githubusercontent.com/{BASE}/st_movenet_lightning_a100_heatmaps_192/st_movenet_lightning_a100_heatmaps_192.keras",
}
out = Path(__file__).resolve().parent.parent / "models_zoo"; out.mkdir(exist_ok=True)
for name, url in FILES.items():
    dst = out / name
    if not dst.exists(): urllib.request.urlretrieve(url, dst)
    print(name, dst.stat().st_size, "bytes")
    assert dst.stat().st_size > 1_000_000, f"{name} looks like a Git-LFS pointer or an error page"
