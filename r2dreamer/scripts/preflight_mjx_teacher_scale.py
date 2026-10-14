"""Bounded, collection-only MJX teacher memory smoke at a chosen world count."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from envs.mjx_teacher_env import MJXTeacherEnv
from training_budget import validate_learning_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--worlds", type=int, required=True)
    parser.add_argument("--contacts-per-world", type=int, default=256)
    args = parser.parse_args()
    if not 2 <= args.worlds <= 1600:
        parser.error("worlds must be between 2 and 1600")
    if args.contacts_per_world < 64:
        parser.error("contacts-per-world must be at least 64")
    capacity = args.worlds * 128
    warmup = validate_learning_budget(args.worlds, 64, 10, capacity, 2949440)
    config = SimpleNamespace(
        task="humanoid_progressive_walking", device="cuda:0",
        mjx_teacher_experimental=True, enable_vision=False,
        scene_path=str(args.scene), env_num=args.worlds, time_limit=4000,
        action_repeat=10, seed=42,
        mjx_contacts_per_world=args.contacts_per_world,
        mjx_arm_action_scale=math.radians(90),
        mjx_leg_action_scale=math.radians(90),
        mjx_other_action_scale=math.radians(90),
        mjx_control_slew_rad_s=2.0,
    )
    print(f"SCALE_PREFLIGHT_BEGIN worlds={args.worlds} "
          f"contacts_per_world={args.contacts_per_world} "
          f"capacity={capacity} conservative_warmup={warmup}", flush=True)
    env = MJXTeacherEnv(config)
    try:
        done = torch.ones(args.worlds, dtype=torch.bool, device="cuda:0")
        action = torch.zeros((args.worlds, 19), device="cuda:0")
        first, _ = env.step(action, done)
        second, _ = env.step(action, torch.zeros_like(done))
        if not torch.isfinite(first["proprio"]).all().item():
            raise RuntimeError("Nonfinite reset observation")
        if not torch.isfinite(second["teacher_state"]).all().item():
            raise RuntimeError("Nonfinite stepped teacher observation")
        if not torch.isfinite(second["reward"]).all().item():
            raise RuntimeError("Nonfinite reward")
        free, total = torch.cuda.mem_get_info()
        print(f"SCALE_PREFLIGHT_OK worlds={args.worlds} "
              f"gpu_used_bytes={total-free} gpu_total_bytes={total}", flush=True)
        print("This does not test replay allocation or optimizer updates.", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
