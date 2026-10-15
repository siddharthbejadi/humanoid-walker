"""Hand crop for the gripper network. The hand centre is not known (MoveNet stops at the wrist), so try several candidate crops beyond the wrist
along the forearm and keep the one with the most skin-coloured pixels (YCrCb), which also rejects red sleeves/cloth."""
import numpy as np, cv2
from pose_arm import RE, RW

def _warp(frame, c, side, out):
    M = np.array([[out / side, 0, out / 2 - c[0] * out / side], [0, out / side, out / 2 - c[1] * out / side]], np.float32)
    return cv2.warpAffine(frame, M, (out, out), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)

def skin_fraction(crop):
    ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb); cr, cb = ycc[..., 1], ycc[..., 2]
    m = (cr > 135) & (cr < 172) & (cb > 85) & (cb < 128) & (ycc[..., 0] > 60)       # skin; saturated red cloth has Cr > 172
    return float(m.mean()), m

def hand_crop(frame, kp, out=96, hand_ratio=0.5, side_ratio=1.8, return_all=False):
    e, w = kp[RE][:2], kp[RW][:2]; f = w - e; lf = float(np.linalg.norm(f)) + 1e-6; d = f / lf
    L = hand_ratio * lf; best = None
    for t in (0.15, 0.5, 0.85):
        for sr in (1.4, 2.0):
            c = w + d * t * L * 1.2; side = max(sr * L, 24.0); crop = _warp(frame, c, side, out); frac, m = skin_fraction(crop)
            yy, xx = np.mgrid[0:out, 0:out]; cen = float((m * np.exp(-(((xx - out / 2) ** 2 + (yy - out / 2) ** 2) / (2 * (out * 0.3) ** 2)))).mean())   # skin near the centre
            score = frac * 0.4 + cen * 1.2
            if best is None or score > best[0]: best = (score, crop, side, frac)
    return (best[1], best[2]) if not return_all else best
