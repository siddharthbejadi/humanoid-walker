"""Preflight for goal_corridor_v3 on the servo scene (run before any paid training).

Checks, in order, and exits non-zero on the first failure:
  1. the servo scene stands with no policy (CPU MuJoCo, 3 s)   -- the soft scene fell in 0.7 s
  2. goal_corridor_v3 runs on the GPU at --worlds and standing starts survive zero
     action until the 5 s stall rule (not a fall); v3 metrics are finite
  3. headless rendering works (exit code 3 if not: train with VIDEO=0)

Usage: python scripts/preflight_servo_v3.py --worlds 64
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCENE = ROOT / "envs/mjx_walking/assets/" / os.environ.get("SCENE", "humanoid_rubber_2mm_servo_v2_kp40.xml")


def fail(msg, code=1):
    print(f"PREFLIGHT FAIL: {msg}", flush=True)
    raise SystemExit(code)


def check_cpu_standing():
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "standing_start")
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.ctrl[:] = d.qpos[m.jnt_qposadr[m.actuator_trnid[:, 0]]]
    for _ in range(int(3. / m.opt.timestep)):
        mujoco.mj_step(m, d)
    if d.qpos[2] < .25:
        fail(f"servo scene does not stand on CPU (torso {d.qpos[2]:.3f} m after 3 s)")
    print(f"[1/3] CPU: servo scene stands with no policy (torso {d.qpos[2]:.3f} m after 3 s)", flush=True)


def check_gpu(worlds):
    import torch
    from hydra import compose, initialize_config_dir

    from envs.mjx_teacher_env import MJXTeacherEnv

    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        cfg = compose("walking_mjx_privileged_teacher_200k", overrides=[
            "logdir=/tmp/preflight", f"env.env_num={worlds}", "env.mjx_reward_profile=goal_corridor_v3",
            "env.time_limit=200000", "env.pose_rsi_json=envs/mjx_walking/assets/manual_rsi_poses_merged.json",
            "env.rsi_probabilities=[1.0,0.0,0.0,0.0]", f"env.scene_path={SCENE}"])
    env = MJXTeacherEnv(cfg.env)
    dev = torch.device("cuda:0")
    zero = torch.zeros(worlds, 19, device=dev)
    env.step(zero, torch.ones(worlds, dtype=torch.bool, device=dev))
    ended = np.full(worlds, np.nan)
    for t in range(300):  # 6 s
        _, last = env.step(zero, torch.zeros(worlds, dtype=torch.bool, device=dev))
        for key in ("reward_track", "reward_landing", "penalty_fall"):
            if not np.all(np.isfinite(np.asarray(env.state.metrics[key]))):
                fail(f"non-finite {key} at step {t}")
        newly = last.cpu().numpy() & np.isnan(ended)
        ended[newly] = (t + 1) * env.raw.control_dt
        if not np.isnan(ended).any():
            break
    env.close()
    early = np.nanmin(ended) if not np.isnan(ended).all() else float("inf")
    if early < 4.5:
        fail(f"a standing world ended at {early:.2f} s with zero action (expected the 5 s stall rule)")
    print(f"[2/3] GPU: {worlds} worlds, goal_corridor_v3 OK; zero-action standing lasts until the stall rule "
          f"(first end {early:.2f} s)", flush=True)


def check_render():
    try:
        import mujoco

        m = mujoco.MjModel.from_xml_path(str(SCENE))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        renderer = mujoco.Renderer(m, 240, 320)
        renderer.update_scene(d)
        frame = renderer.render()
        renderer.close()
        if frame.mean() < 1:
            raise RuntimeError("blank frame")
    except Exception as exc:
        fail(f"rendering with MUJOCO_GL={os.environ.get('MUJOCO_GL')} failed ({type(exc).__name__}: {exc}); "
             "train with VIDEO=0 or install libegl1/libosmesa6", code=3)
    print(f"[3/3] Rendering OK (MUJOCO_GL={os.environ.get('MUJOCO_GL')})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=64)
    args = parser.parse_args()
    check_cpu_standing()
    check_gpu(args.worlds)
    check_render()
    print("PREFLIGHT PASSED", flush=True)
