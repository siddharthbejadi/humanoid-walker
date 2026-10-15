#!/usr/bin/env python3
"""Export the student to ONNX (float) and int8 QDQ, then check float-vs-int8 parity on real frames.

    ~/hc_train/bin/python -I export_onnx.py CKPT.pt CACHE.npz OUT_DIR [--size 224]
Input: float32 NCHW RGB 0..255 (normalisation is inside the graph).  Output: [cx, cy, size, grip].
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hc_model import HandNet
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", type=Path); ap.add_argument("cache", type=Path); ap.add_argument("out", type=Path)
ap.add_argument("--size", type=int, default=224); ap.add_argument("--calib", type=int, default=300)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)

net = HandNet(pretrained=False)
sd = torch.load(a.ckpt, map_location="cpu", weights_only=False); sd = sd["state_dict"]
net.load_state_dict(sd); net.eval()
f32 = a.out / "student_float.onnx"
torch.onnx.export(net, torch.zeros(1, 3, a.size, a.size), f32, input_names=["image"], output_names=["targets"], opset_version=13, dynamo=False)

d = np.load(a.cache)
imgs, tg = d["images"], d["targets"]
rng = np.random.default_rng(0)
cal_idx = rng.choice(len(imgs), a.calib, replace=False)
prep = lambda i: imgs[i].astype(np.float32).transpose(2, 0, 1)[None]

class Reader(CalibrationDataReader):
    def __init__(s): s.it = iter(cal_idx)
    def get_next(s):
        i = next(s.it, None)
        return None if i is None else {"image": prep(i)}

i8 = a.out / "student_int8_qdq.onnx"
quantize_static(str(f32), str(i8), Reader(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8, per_channel=True)

test = rng.choice(len(imgs), 400, replace=False)
res = {}
for name, p in (("float", f32), ("int8", i8)):
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    res[name] = np.concatenate([s.run(None, {"image": prep(i)})[0] for i in test])
ref = tg[test]
rep = {"float_vs_int8_mae": dict(zip(["cx", "cy", "size", "grip"], np.abs(res["float"] - res["int8"]).mean(0).round(4).tolist())),
       "float_vs_label_mae": np.abs(res["float"] - ref).mean(0).round(4).tolist(),
       "int8_vs_label_mae": np.abs(res["int8"] - ref).mean(0).round(4).tolist(),
       "note": "frames sampled from the training cache (parity check, not generalisation)",
       "float_mb": round(f32.stat().st_size / 1e6, 2), "int8_mb": round(i8.stat().st_size / 1e6, 2)}
(a.out / "parity.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
