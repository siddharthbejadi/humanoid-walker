"""Replay a keypoint recording through the whole retargeting chain and render the simulated humanoid. No camera, no images.

    python replay.py                                   # the bundled sample -> replay_out.mp4 (robot + skeleton panels)
    python replay.py data/sample/live_keypoints.json --out my.mp4 --show --smooth 0.7

Chain: keypoints -> plausibility gate -> reachability-aware IK (+ depth cue) -> One-Euro filter -> MuJoCo humanoid right arm. Prints how many frames passed the gate."""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from armcopy.pose import arm_vectors, forward_fractions, RS, RE, RW
from armcopy.sim import Humanoid, ArmIK3
from armcopy.filters import OneEuro


def retarget_sequence(keypoints, frame_size, smooth=0.7, fps=15.0):
    """-> (angles (N,3) [shoulder, arm, elbow] in rad, found (N,) bool). Frames that fail the gate hold, then ease to the rest pose after ~1 s."""
    robot = Humanoid(32, 32); ik = ArmIK3(robot); filt = [OneEuro(smooth) for _ in range(3)]; q = ik.REST.copy(); miss = 0; out, found = [], []
    for k, kp in enumerate(keypoints):
        v = arm_vectors(np.asarray(kp, np.float32), size=tuple(frame_size))
        if v is not None:
            qn, _ = ik.fit(*v, *forward_fractions(np.asarray(kp, np.float32))); q = np.array([f(x, k / fps) for f, x in zip(filt, qn)]); miss = 0
        else:
            miss += 1
            if miss > int(fps):
                q = 0.9 * q + 0.1 * ik.REST
        out.append(q.copy()); found.append(v is not None)
    return np.array(out), np.array(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keypoints", nargs="?", type=Path, default=Path(__file__).resolve().parent / "data" / "sample" / "live_keypoints.json")
    ap.add_argument("--out", default="replay_out.mp4"); ap.add_argument("--smooth", type=float, default=0.7); ap.add_argument("--show", action="store_true"); ap.add_argument("--fps", type=float, default=15.0)
    a = ap.parse_args()
    d = json.loads(a.keypoints.read_text()); kps, size, src_fps = d["keypoints"], d["frame_size"], d["fps"]
    n_out = int(len(kps) * a.fps / src_fps); idx = [min(int(k * src_fps / a.fps), len(kps) - 1) for k in range(n_out)]      # resample to the output rate
    q, found = retarget_sequence([kps[i] for i in idx], size, a.smooth, a.fps)
    robot = Humanoid(480, 480); H, W = 480, int(480 * size[0] / size[1]); w = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W + 480, H))
    for k, i in enumerate(idx):
        kp = np.asarray(kps[i], np.float32); skel = np.full((H, W, 3), 32, np.uint8); s = H / size[1]; p = (kp[:, :2] * s).astype(int)
        for j in range(17): cv2.circle(skel, tuple(p[j]), 3, (110, 110, 110), -1)
        if found[k]:
            for a_, b_ in ((RS, RE), (RE, RW)): cv2.line(skel, tuple(p[a_]), tuple(p[b_]), (0, 220, 0), 4, cv2.LINE_AA)
            for j in (RS, RE, RW): cv2.circle(skel, tuple(p[j]), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(skel, "MoveNet keypoints (no image)" + ("" if found[k] else "  - arm lost, easing to rest"), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        sim = cv2.cvtColor(robot.render([*q[k], 0.0]), cv2.COLOR_RGB2BGR); frame = np.hstack([skel, sim]); w.write(frame)
        if a.show:
            cv2.imshow("replay (q quits)", frame)
            if cv2.waitKey(int(1000 / a.fps)) & 0xFF == ord("q"): break
    w.release(); print(f"{a.keypoints.name}: {len(kps)} frames at {src_fps} fps, gate passed {100 * found.mean():.1f}% -> {a.out} ({n_out / a.fps:.1f} s)")


if __name__ == "__main__":
    main()
