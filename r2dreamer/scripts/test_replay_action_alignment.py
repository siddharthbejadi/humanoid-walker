"""Real replay causality checks, including the CUDA path used for training."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from buffer import Buffer


def populated_buffer(device, batch_length=64):
    buffer = Buffer(SimpleNamespace(
        device=device, storage_device=device, batch_size=16,
        batch_length=batch_length, max_size=32768,
        cache_trajectories=True, store_log_metrics=False))
    worlds = torch.arange(16, device=device)
    channels = torch.arange(19, device=device, dtype=torch.float32)
    for time in range(batch_length + 10):
        marker = (1000 * worlds + time).float()
        action = marker[:, None] * 100 + channels
        buffer.add_transition(TensorDict({
            "proprio": marker[:, None].expand(-1, 45).clone(),
            "action": action,
            "stoch": marker[:, None, None].expand(-1, 2, 3).clone(),
            "deter": marker[:, None].expand(-1, 4).clone(),
            "episode": worlds,
            "is_first": torch.full((16, 1), time % 17 == 0,
                                   dtype=torch.bool, device=device),
        }, batch_size=(16,)))
    return buffer


def check_previous_actions_and_observations_keep_exact_causal_alignment(device):
    torch.manual_seed(29)
    buffer = populated_buffer(device)
    storage = buffer._buffer._storage._storage
    source = storage.clone()
    channels = torch.arange(19, device=device, dtype=torch.float32)
    for _ in range(5):
        data, indices, initial = buffer.sample()
        assert tuple(data["action"].shape) == (16, 64, 19)
        current = data["proprio"][..., 0]
        expected = (current - 1)[..., None] * 100 + channels
        assert torch.equal(data["action"], expected)
        assert torch.equal(data["proprio"], current[..., None].expand(-1, -1, 45))
        assert torch.equal(data["is_first"][..., 0], current.remainder(1000).remainder(17) == 0)
        assert torch.equal(initial[0], (current[:, 0] - 1)[:, None, None].expand(-1, 2, 3))
        assert torch.equal(initial[1], (current[:, 0] - 1)[:, None].expand(-1, 4))
        rows, worlds = indices
        assert torch.equal(current, source["proprio"][rows, worlds, 0])
    for key in source.keys():
        assert torch.equal(storage[key], source[key]), f"Sampling modified stored {key}"


def check_recomputed_latents_update_only_the_sampled_storage_indices(device):
    torch.manual_seed(31)
    buffer = populated_buffer(device)
    data, indices, _ = buffer.sample()
    storage = buffer._buffer._storage._storage
    source = storage.clone()
    rows, worlds = (index.flatten() for index in indices)
    stoch = torch.full_like(data["stoch"], -101)
    deter = torch.full_like(data["deter"], -202)
    expected_stoch = source["stoch"].clone()
    expected_deter = source["deter"].clone()
    expected_stoch[rows, worlds] = stoch.flatten(0, 1)
    expected_deter[rows, worlds] = deter.flatten(0, 1)
    buffer.update(indices, stoch, deter)
    assert torch.equal(storage["stoch"], expected_stoch)
    assert torch.equal(storage["deter"], expected_deter)
    for key in ("proprio", "action", "is_first", "episode"):
        assert torch.equal(storage[key], source[key]), f"Latent update modified {key}"


class CPUReplayActionAlignmentTests(unittest.TestCase):
    device = "cpu"

    def test_previous_actions_and_observations_keep_exact_causal_alignment(self):
        check_previous_actions_and_observations_keep_exact_causal_alignment(self.device)

    def test_recomputed_latents_update_only_the_sampled_storage_indices(self):
        check_recomputed_latents_update_only_the_sampled_storage_indices(self.device)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available on this host")
class CUDAReplayActionAlignmentTests(CPUReplayActionAlignmentTests):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
