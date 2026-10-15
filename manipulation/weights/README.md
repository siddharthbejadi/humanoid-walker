# Weights

| File | What | Trained on | Use |
|---|---|---|---|
| `grip_v2.onnx` (8.9 MB) | gripper network: 96x96 hand crop -> open/closed logit (MobileNetV2) | 6000 FreiHAND images, 40 epochs (`models/train_grip.py`) | **research / non-commercial only** (inherits FreiHAND's terms) |

Not here: the MoveNet files (ST model zoo, SLA0044): `python models/fetch_zoo_models.py` downloads the original. My fine-tuned MoveNet is a derivative of that model and is not redistributed;
`models/finetune_movenet.py` + `models/export_movenet_int8.py` rebuild it from your own recordings. `demo.py` uses `weights/grip_v2.onnx` if present.
