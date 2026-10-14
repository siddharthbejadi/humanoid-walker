"""Checkpoint-only recurrent policy inference; no dataset or gait-file dependency."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from omegaconf import OmegaConf
from tensordict import TensorDict

from dreamer import Dreamer
from script_free_contract import validate_student_environment


def _devices(value, device):
    if isinstance(value, dict):
        return {key: device if key == "device" else _devices(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_devices(item, device) for item in value]
    return value


class ScriptFreePolicy:
    def __init__(self, checkpoint, device="cpu"):
        checkpoint = (torch.load(Path(checkpoint), map_location=device, weights_only=False)
                      if isinstance(checkpoint, (str, Path)) else checkpoint)
        self.contract = validate_student_environment(checkpoint.get("training_contract"))
        metadata = checkpoint.get("imitation")
        if (not isinstance(metadata, dict) or metadata.get("version") != 1 or
                metadata.get("mode") != "full_command_bc" or
                type(metadata.get("pretraining_updates")) is not int or metadata["pretraining_updates"] <= 0):
            raise ValueError("Checkpoint is not a verified full-command imitation student")
        model = deepcopy(checkpoint.get("model_config"))
        if not isinstance(model, dict) or model.get("encoder", {}).get("mlp_keys") != "^proprio$":
            raise ValueError("Checkpoint needs its original sensor-only model configuration")
        if model.get("encoder", {}).get("cnn_keys") != "$^":
            raise ValueError("Student inference must not require a camera or extra encoder input")
        if model.get("auxiliary_state", False):
            raise ValueError("This student recipe excludes privileged auxiliary targets")
        self.device = torch.device(device)
        config = OmegaConf.create(_devices(model, str(self.device)))
        obs = spaces.Dict({"proprio": spaces.Box(-np.inf, np.inf, (45,), np.float32)})
        actions = spaces.Box(-1., 1., (19,), np.float32)
        self.agent = Dreamer(config, obs, actions).to(self.device)
        self.agent.load_state_dict(checkpoint["agent_state_dict"], strict=True)
        self.agent.clone_and_freeze()
        self.agent.eval()
        if not all(bool(torch.isfinite(value).all()) for value in self.agent.state_dict().values()):
            raise ValueError("Student checkpoint contains nonfinite weights")
        self.agent.export_model_config = model
        self.imitation = metadata
        self.state = None

    def reset(self, worlds=1):
        if type(worlds) is not int or worlds < 1:
            raise ValueError("Policy reset needs a positive batch size")
        self.state = self.agent.get_initial_state(worlds)

    @torch.no_grad()
    def act(self, proprio, is_first=None):
        proprio = torch.as_tensor(proprio, device=self.device, dtype=torch.float32)
        if proprio.ndim != 2 or proprio.shape[1] != 45 or not bool(torch.isfinite(proprio).all()):
            raise ValueError("Policy input must be finite worlds-by-45 sensor-style observations")
        worlds = proprio.shape[0]
        if self.state is None:
            self.reset(worlds)
            if is_first is None:
                is_first = torch.ones(worlds, 1, dtype=torch.bool, device=self.device)
        elif self.state.batch_size != torch.Size([worlds]):
            raise ValueError("Reset recurrent policy state before changing the world count")
        if is_first is None:
            is_first = torch.zeros(worlds, 1, dtype=torch.bool, device=self.device)
        is_first = torch.as_tensor(is_first, device=self.device, dtype=torch.bool)
        if is_first.shape != (worlds, 1):
            raise ValueError("is_first must have shape worlds-by-1")
        observation = TensorDict({"proprio": proprio, "is_first": is_first}, batch_size=[worlds])
        action, self.state = self.agent.act(observation, self.state, eval=True)
        if action.shape != (worlds, 19) or not bool(torch.isfinite(action).all()) or bool((action.abs() > 1).any()):
            raise RuntimeError("Student produced invalid full motor actions")
        return action
