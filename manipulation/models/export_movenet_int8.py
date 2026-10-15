"""Fine-tuned MoveNet (.keras) -> int8 TFLite (uint8 image in, int8 heatmaps out, same I/O as the zoo model) + parity vs float on held-out frames.
    python export_movenet_int8.py MODEL.keras CACHE.npz OUT.tflite"""
import sys, numpy as np, tensorflow as tf, keras
mk, cache, out = sys.argv[1:4]
m = keras.models.load_model(mk, compile=False)
d = np.load(cache); X, cid, names = d["images"], d["clip_id"], list(d["clip_names"])
rng = np.random.default_rng(0); cal = X[rng.choice(len(X), 300, replace=False)]
def rep():
    for im in cal: yield [im[None].astype(np.float32) / 127.5 - 1.0]
conv = tf.lite.TFLiteConverter.from_keras_model(m)
conv.optimizations = [tf.lite.Optimize.DEFAULT]; conv.representative_dataset = rep
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]; conv.inference_input_type = tf.uint8; conv.inference_output_type = tf.int8
open(out, "wb").write(conv.convert())
it = tf.lite.Interpreter(model_path=out); it.allocate_tensors(); i, o = it.get_input_details()[0], it.get_output_details()[0]
print("in", i["shape"], i["dtype"].__name__, i["quantization"], "out", o["shape"], o["dtype"].__name__, o["quantization"])
idx = np.flatnonzero(cid == names.index("take3")); vi = idx[int(len(idx) * 0.8):][::3]
def kp(h):
    r = []
    for ch in (5, 6, 8, 10):
        y, x = np.unravel_index(h[..., ch].argmax(), h.shape[:2]); r.append((x, y))
    return np.array(r, float)
err = []
for n in vi:
    f = m(X[n][None].astype(np.float32) / 127.5 - 1.0, training=False).numpy()[0]
    it.set_tensor(i["index"], X[n][None].astype(np.uint8)); it.invoke()
    q = it.get_tensor(o["index"])[0].astype(np.float32); q = (q - o["quantization"][1]) * o["quantization"][0]
    err.append(np.hypot(*(kp(f) - kp(q)).T))
err = np.array(err); print("int8 vs float keypoint distance (cells; 1 cell = 13 px at 640) mean per kp [Lsh Rsh Rel Rwr]:", err.mean(0).round(2), "| frames with all <=1 cell:", round(100 * float((err <= 1).all(1).mean()), 1), "%")
