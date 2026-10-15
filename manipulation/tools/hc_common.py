"""Shared helpers: hand landmarks -> 4 control targets, and the hand -> arm retargeting map.

Targets per frame (all derived from MediaPipe's 21 hand landmarks, image coordinates x right, y down):
    cx, cy : palm centre in the image, 0..1
    size   : wrist -> middle-finger-knuckle length / image diagonal  (bigger = hand closer to the camera)
    grip   : finger curl, 0 = open hand, 1 = fist  (tip-to-wrist distance relative to knuckle-to-wrist distance)
"""
import numpy as np

PALM = [0, 5, 9, 13, 17]
TIPS, MCPS = [8, 12, 16, 20], [5, 9, 13, 17]
NAMES = ["cx", "cy", "size", "grip"]


def targets_from_landmarks(lm, width, height):
    """lm: (21, 3) normalised landmarks -> np.array([cx, cy, size, grip])."""
    px = lm[:, :2] * np.array([width, height])
    diag = float(np.hypot(width, height))
    centre = lm[PALM, :2].mean(0)
    # palm size = mean of palm length (wrist -> middle knuckle) and palm width (index knuckle -> pinky knuckle):
    # steadier than one segment, and unchanged by curling the fingers (so it measures depth, not grip)
    size = 0.5 * (np.linalg.norm(px[9] - px[0]) + np.linalg.norm(px[5] - px[17])) / diag
    reach = np.linalg.norm(px[TIPS] - px[0], axis=1) / np.maximum(np.linalg.norm(px[MCPS] - px[0], axis=1), 1e-6)
    openness = float(reach.mean())                       # ~1.8-2.0 open hand, ~0.9-1.2 fist
    grip = float(np.clip((1.9 - openness) / (1.9 - 1.0), 0.0, 1.0))
    return np.array([centre[0], centre[1], size, grip], np.float32)


def fill_short_gaps(values, valid, max_gap=5):
    """Linear interpolation over gaps of at most `max_gap` frames; returns (values, valid, interpolated_mask)."""
    values, valid = values.copy(), valid.copy()
    interp = np.zeros(len(valid), bool)
    idx = np.flatnonzero(valid)
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                values[a + k] = (1 - t) * values[a] + t * values[b]
            valid[a + 1:b] = True
            interp[a + 1:b] = True
    return values, valid, interp


def smooth(values, valid, width=3):
    """Centred moving average over valid neighbours (reduces teacher jitter)."""
    out = values.copy()
    half = width // 2
    for i in np.flatnonzero(valid):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        sel = valid[lo:hi]
        out[i] = values[lo:hi][sel].mean(0)
    return out


# ---- hand -> robot right arm (calibrated linear map; the same numbers can drive the real servos later) ----
# joint order: shoulder (pitch), arm (lateral), elbow, gripper.  Ranges are inside the simulator's +-pi/2 limits.
ARM_MAP = dict(
    cx=(0.10, 0.90), cy=(0.10, 0.90), size=(0.03, 0.15),      # observed hand ranges (input side, from the recordings)
    shoulder=(1.20, -0.40),                                   # hand high in image -> arm raised
    arm=(-0.80, 0.80),                                        # hand left/right -> arm swings sideways
    elbow=(0.0, 1.30),                                        # hand big (close) -> elbow extended
    grip=(0.0, 1.0),                                          # fist -> gripper closed
)


def lin(x, src, dst):
    t = np.clip((x - src[0]) / (src[1] - src[0]), 0.0, 1.0)
    return dst[0] + t * (dst[1] - dst[0])


def retarget(targets):
    """(N,4) [cx, cy, size, grip] -> (N,4) joint angles [shoulder, arm, elbow, gripper] in rad."""
    t = np.atleast_2d(targets)
    m = ARM_MAP
    return np.stack([lin(t[:, 1], m["cy"], m["shoulder"]), lin(t[:, 0], m["cx"], m["arm"]),
                     lin(t[:, 2], m["size"], m["elbow"]), lin(t[:, 3], (0.0, 1.0), m["grip"])], axis=1).astype(np.float32)
