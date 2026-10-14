#!/usr/bin/env python3
"""Run the walking student in the project's simulator (same settings it was tested in) and show it in MuJoCo.

    ~/r2dreamer/.venv/bin/python watch_policy.py --repo ~/sf_v6/r2dreamer \
        --checkpoint student_realistic_12k.pt --seconds 20 --seed 1100 [--video walk.mp4] [--view]

Step 1 simulates 1 robot on the GPU (MJX-Warp) and saves the trajectory (--out walk.npz).
Step 2 (--view) replays it at real time in an interactive MuJoCo viewer window, looping until you close it.
--video writes an MP4 with an off-screen renderer (may fail on machines without working EGL/OSMesa).
Nothing here talks to the robot.
"""
import argparse
import os
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="source tree with script_free_policy.py (source_v6)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seconds", type=float, default=20.)
    ap.add_argument("--seed", type=int, default=1100)
    ap.add_argument("--out", type=Path, default=Path("walk.npz"))
    ap.add_argument("--video", type=Path)
    ap.add_argument("--view", action="store_true", help="replay in an interactive MuJoCo window")
    ap.add_argument("--replay", type=Path, help="skip simulating; replay an existing .npz")
    args = ap.parse_args()
    repo = Path(args.repo).expanduser().resolve()
    args.checkpoint = Path(args.checkpoint).expanduser().resolve()   # resolve BEFORE changing directory
    args.out = args.out.expanduser().resolve()
    args.video = args.video.expanduser().resolve() if args.video else None
    args.replay = args.replay.expanduser().resolve() if args.replay else None
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    import numpy as np
    import hydra
    import mujoco

    if args.replay is None:
        import torch
        from scripts.collect_script_free_demonstrations import GPUStudentProvider
        from script_free_policy import ScriptFreePolicy
        from training_contract import check_checkpoint_contract

        with hydra.initialize_config_dir(version_base=None, config_dir=str(repo / "configs")):
            config = hydra.compose("walking_mjx_script_free", overrides=[f"seed={args.seed}"])
        config.env.env_num = 1
        config.env.eval_episode_num = 0
        policy = ScriptFreePolicy(Path(args.checkpoint).expanduser().resolve(), device="cuda:0")
        provider = GPUStudentProvider(config, 1)
        check_checkpoint_contract({"training_contract": policy.contract}, provider.contract, mode="resume")
        print("checkpoint matches this simulator's contract; simulating...", flush=True)

        first = provider.reset(args.seed)
        qpos = [provider.audit()["qpos"][0].copy()]
        proprio, steps = first["proprio"], int(round(args.seconds / .02))
        policy.reset(1)
        valid = {"valid_left_step": 0., "valid_right_step": 0., "alternating_steps": 0.}
        fell, t0 = False, time.time()
        for step in range(steps):
            action = policy.act(proprio).cpu().numpy()
            out = provider.step(action)
            proprio = out["proprio"]
            qpos.append(out["audit"]["qpos"][0].copy())
            for key in valid:
                valid[key] += float(out["logs"][key].reshape(-1)[0])
            if bool(out["causes"]["physical_fall"].any()):
                fell = True
            if bool(out["ended"].any()):
                break
        qpos = np.array(qpos)
        dy = float(qpos[-1, 1] - qpos[0, 1])
        print(f"simulated {len(qpos) - 1} control steps ({(len(qpos) - 1) * .02:.1f} s) in {time.time() - t0:.0f} s wall")
        print(f"forward travel {-dy:.2f} m | fell: {fell} | valid steps L {valid['valid_left_step']:.0f} "
              f"R {valid['valid_right_step']:.0f} | alternating {valid['alternating_steps']:.0f} | "
              f"final height {qpos[-1, 2]:.3f} m")
        model_path = None
        model = provider.env.raw.m
        np.savez(args.out, qpos=qpos, dt=.02)
        print("saved", args.out)
        # keep a CPU copy of the model next to the trajectory so --replay works without the GPU
        mujoco.mj_saveLastXML(str(args.out.with_suffix(".xml")), model)
        provider.close()
    else:
        qpos = np.load(args.replay)["qpos"]
        # The saved scene refers to meshes by relative path (meshdir="meshes/"); supply them explicitly.
        meshes = repo / "envs" / "mjx_walking" / "assets" / "meshes"
        assets = {f"meshes/{path.name}": path.read_bytes() for path in meshes.iterdir() if path.is_file()}
        model = mujoco.MjModel.from_xml_string(args.replay.with_suffix(".xml").read_text(), assets)

    data = mujoco.MjData(model)

    def pose(index):
        data.qpos[:] = qpos[index]
        mujoco.mj_forward(model, data)

    if args.video:
        try:
            import imageio.v2 as imageio
            renderer = mujoco.Renderer(model, 480, 640)
            camera = mujoco.MjvCamera()
            camera.type, camera.distance, camera.azimuth, camera.elevation = mujoco.mjtCamera.mjCAMERA_FREE, 1.1, 150., -12.
            with imageio.get_writer(args.video, fps=25) as writer:
                for i in range(0, len(qpos), 2):
                    pose(i)
                    camera.lookat[:] = data.qpos[:3]
                    renderer.update_scene(data, camera)
                    writer.append_data(renderer.render())
            print("wrote video", args.video)
        except Exception as error:   # EGL/OSMesa problems are common on WSL
            print("VIDEO FAILED (", type(error).__name__, error, ") - use --view instead")

    if args.view:
        import mujoco.viewer
        print("opening viewer; close the window to quit")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.1, 150., -12.
            while viewer.is_running():
                start = time.time()
                for i in range(len(qpos)):
                    if not viewer.is_running():
                        break
                    pose(i)
                    viewer.cam.lookat[:] = data.qpos[:3]
                    viewer.sync()
                    time.sleep(max(0., .02 - (time.time() - start - i * .02) * 0 - 0.0))
                time.sleep(1.0)


if __name__ == "__main__":
    main()
