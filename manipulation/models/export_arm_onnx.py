"""Float ONNX of the joint-space student (local test only).  export_arm_onnx.py CKPT.pt CACHE.npz OUT.onnx
Input float32 NCHW RGB 0..255, 192x192 (padded square).  Output [shoulder, arm, elbow, present_logit] in real units."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armcopy.nets import HandNet
import onnxruntime as ort
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
net = HandNet(pretrained=False); net.load_state_dict(ck["state_dict"]); net.eval()
torch.onnx.export(net, torch.zeros(1, 3, 192, 192), sys.argv[3], input_names=["image"], output_names=["out"], opset_version=13, dynamo=False)
X = np.load(sys.argv[2])["images"][::97][:40]
x = torch.from_numpy(X).permute(0, 3, 1, 2).float()
with torch.no_grad(): ref = net(x).numpy()
got = np.concatenate([ort.InferenceSession(sys.argv[3], providers=["CPUExecutionProvider"]).run(None, {"image": x[i:i + 1].numpy()})[0] for i in range(len(x))])
print("onnx vs torch max abs diff:", float(np.abs(ref - got).max()), "on", len(x), "frames")
