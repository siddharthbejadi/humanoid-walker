"""Live view of a running MJX training: all worlds in one MuJoCo window.

The trainer writes <rundir>/live_view.npz ~10x per second (qpos of every
world). This draws world 0 as the main robot and the others next to it in a
grid. Robots only move while training collects experience; during optimizer
updates the trainer is busy learning, so they freeze for a moment.

Usage: python scripts/watch_training.py --rundir /path/to/run
"""
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCENE = Path(__file__).resolve().parent.parent / "envs/mjx_walking/assets/humanoid_rubber_2mm_candidate.xml"
SPACING = .6  # metres between robots in the grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rundir", required=True)
    args = parser.parse_args()
    live = Path(args.rundir) / "live_view.npz"
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    main_data = mujoco.MjData(model)
    others, last_mtime, last_print = [], 0., 0.
    print(f"Waiting for {live} (written by the trainer while it collects experience)...")
    with mujoco.viewer.launch_passive(model, main_data) as viewer:
        while viewer.is_running():
            try:
                mtime = live.stat().st_mtime
            except FileNotFoundError:
                mtime = last_mtime
            if mtime != last_mtime:
                try:
                    snap = np.load(live)
                    qpos, reward, step = snap["qpos"], snap["reward"], snap["step"]
                except Exception:
                    # On /mnt/c the trainer's rename briefly leaves no/partial file; try next frame.
                    viewer.sync()
                    time.sleep(.02)
                    continue
                last_mtime = mtime
                while len(others) < len(qpos) - 1:
                    others.append(mujoco.MjData(model))
                with viewer.lock():
                    for k, data in enumerate([main_data] + others[:len(qpos) - 1]):
                        data.qpos[:] = qpos[k]
                        # Each world walks from its own origin; lay them out in a 4-wide grid.
                        data.qpos[0] += (k % 4) * SPACING
                        data.qpos[1] += (k // 4) * SPACING
                        mujoco.mj_forward(model, data)
                    viewer.user_scn.ngeom = 0
                    for data in others[:len(qpos) - 1]:
                        mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb,
                                            mujoco.mjtCatBit.mjCAT_DYNAMIC, viewer.user_scn)
                if time.time() - last_print > .5:
                    print("\rreward/world " + " ".join(f"{r:+6.1f}" for r in reward)
                          + "  episode step " + " ".join(f"{int(s):4d}" for s in step), end="", flush=True)
                    last_print = time.time()
            viewer.sync()
            time.sleep(.02)


if __name__ == "__main__":
    main()
