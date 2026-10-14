"""Check that a GPU teacher state produces a real external-view MP4."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import torch

from envs.mjx_teacher_env import MJXTeacherEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    config = SimpleNamespace(
        task="humanoid_progressive_walking", device="cuda:0",
        mjx_teacher_experimental=True, enable_vision=False,
        scene_path=str(args.scene), env_num=2, time_limit=4000,
        action_repeat=10, seed=42, mjx_contacts_per_world=256,
        mjx_arm_action_scale=math.radians(90),
        mjx_leg_action_scale=math.radians(90),
        mjx_other_action_scale=math.radians(90),
        mjx_control_slew_rad_s=2.0,
    )
    env = MJXTeacherEnv(config, evaluation=True)
    try:
        action = torch.zeros((2, 19), device="cuda:0")
        done = torch.ones(2, dtype=torch.bool, device="cuda:0")
        frames = []
        for index in range(3):
            env.step(action, done if index == 0 else torch.zeros_like(done))
            result = env.render_first(width=320, height=320)
            if result["error"]:
                raise RuntimeError(result["error"])
            frame = result["frame"]
            if frame.shape != (320, 320, 3):
                raise RuntimeError(f"Unexpected frame shape {frame.shape}")
            frames.append(frame)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.output, frames, fps=25, codec="libx264",
                        quality=7, macro_block_size=None)
        if args.output.stat().st_size < 1000:
            raise RuntimeError("Video output is unexpectedly small")
        print(f"MJX_TEACHER_VIDEO_OK {args.output} frames={len(frames)} "
              f"bytes={args.output.stat().st_size}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
