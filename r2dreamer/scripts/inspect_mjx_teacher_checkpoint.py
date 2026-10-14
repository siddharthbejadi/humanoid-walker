"""Read-only checkpoint metadata and loadability check."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    path = args.checkpoint.resolve(strict=True)
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {"step", "optimizer_updates", "agent_state_dict", "optims_state_dict"}
    if not required.issubset(state):
        raise ValueError(f"Missing checkpoint keys: {sorted(required - state.keys())}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"CHECKPOINT_OK path={path} step={int(state['step'])} "
          f"optimizer_updates={int(state['optimizer_updates'])} "
          f"bytes={path.stat().st_size} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
