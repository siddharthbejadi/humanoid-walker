"""ST model-zoo MoveNet (int8 heatmaps) -> person's right-arm keypoints -> humanoid right-arm joint angles.

COCO keypoints used: 6 right shoulder, 8 right elbow, 10 right wrist (the PERSON's right; for a camera facing the person it is on the image left).
Only 2D image-plane angles are available, so the arm swing is in the image plane (abduction + elbow flexion); gripper is not observable here.
"""
from pathlib import Path
import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models_zoo"
RS, RE, RW = 6, 8, 10


class MoveNet:
    def __init__(self, res=192, name=None):
        from ai_edge_litert.interpreter import Interpreter           # lazy: the gate / geometry below need only numpy
        self.res = res                                # name: e.g. "st_movenet_ft_192_int8.tflite" (the fine-tuned student)
        self.it = Interpreter(model_path=str(MODELS / (name or f"st_movenet_{res}_int8.tflite")), num_threads=4)
        self.it.allocate_tensors()
        self.inp, self.out = self.it.get_input_details()[0], self.it.get_output_details()[0]

    def __call__(self, frame_bgr):
        """-> (17, 3) keypoints [x, y, score] in ORIGINAL frame pixels. Frame is padded (not stretched) to a square."""
        h, w = frame_bgr.shape[:2]
        s = max(h, w)
        sq = np.zeros((s, s, 3), np.uint8)
        sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = frame_bgr[:, :, ::-1]
        import cv2
        x = cv2.resize(sq, (self.res, self.res), interpolation=cv2.INTER_AREA)[None]
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        q = self.it.get_tensor(self.out["index"])[0].astype(np.float32)
        hm = (q - self.out["quantization"][1]) * self.out["quantization"][0]            # (H, W, 17)
        H, W, K = hm.shape
        flat = hm.reshape(-1, K)
        idx = flat.argmax(0)
        ys, xs = idx // W, idx % W
        kp = np.zeros((K, 3), np.float32)
        kp[:, 0] = ((xs + 0.5) / W * s) - (s - w) / 2
        kp[:, 1] = ((ys + 0.5) / H * s) - (s - h) / 2
        kp[:, 2] = flat.max(0)
        return kp


U_MIN, F_MIN = 0.4, 0.3          # shortest plausible upper arm / forearm, in shoulder widths (foreshortened arms are short)
EDGE = 0.04     # keypoints this close to the image border (fraction of the padded square) mean the body is cut off: reject


def arm_vectors(kp, min_score=0.04, size=None):      # 0.04: this model's heatmap peaks are low (median ~0.08); position is usually still right
    """-> (upper, fore) 2D vectors in the (outward, up) plane for the person's right arm, or None if the detection is not plausible.
    Camera faces the person, so the person's right arm points to image-left when it goes outward.
    Plausibility (rejects the junk detections seen when the body is cut off): both shoulders found and roughly level, and the
    upper arm / forearm lengths are sensible relative to the shoulder width."""
    sc, e, w, ls = kp[RS], kp[RE], kp[RW], kp[5]
    if min(sc[2], e[2], w[2], ls[2]) < min_score:
        return None
    if size is not None:                                    # (width, height) of the frame: cut-off body -> junk keypoints on the border
        m = EDGE * max(size)
        if any(p[0] < m or p[0] > size[0] - m or p[1] < m or p[1] > size[1] - m for p in (sc, e, ls)):         # wrist excluded: an outstretched arm legitimately reaches the border
            return None
    width = abs(sc[0] - ls[0])
    if width < 20 or abs(sc[1] - ls[1]) > 0.6 * width:
        return None
    if size is not None and width > 0.35 * size[0]:       # person too close to the lens (close-up): skeleton is junk
        return None
    u, f = e[:2] - sc[:2], w[:2] - e[:2]
    lu, lf = np.linalg.norm(u), np.linalg.norm(f)
    if not (U_MIN * width < lu < 2.6 * width and F_MIN * width < lf < 2.6 * width):
        return None
    return np.array([-u[0], -u[1]]), np.array([-f[0], -f[1]])


K_UPPER, K_FORE = 1.1, 0.75        # typical in-plane upper arm / forearm length in shoulder widths (take 3 medians 1.05 / 0.69)
DEADBAND = 0.8                     # ratio above this = in the image plane (perspective + noise), no forward component


def forward_fractions(kp):
    """Foreshortening -> how much of each segment points toward the camera, in [0, 1]. 0 = lies in the image plane."""
    W = abs(kp[RS][0] - kp[5][0])
    lu, lf = np.linalg.norm(kp[RE][:2] - kp[RS][:2]), np.linalg.norm(kp[RW][:2] - kp[RE][:2])
    fr = lambda l, k: 0.0 if l / (k * W) > DEADBAND else float(np.sqrt(max(0.0, 1 - (l / (k * W) / DEADBAND) ** 2)))
    return fr(lu, K_UPPER), fr(lf, K_FORE)
