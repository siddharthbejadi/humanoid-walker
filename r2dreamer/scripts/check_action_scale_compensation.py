"""Numeric check of _apply_action_scale_compensation (no GPU/MJX needed).

Verifies the claim made to the user: after rescaling the actor's mean head
for an action_scale change, the EFFECTIVE PHYSICAL COMMAND
(action_scale * tanh(mean)) stays approximately continuous across the
transition, for inputs in the policy's actually-observed operating range
(std ~0.2-0.3, confirmed from this project's own training logs). It also
demonstrates the known limitation: large pre-tanh means do NOT compensate
exactly, since tanh saturates nonlinearly.

Run: python scripts/check_action_scale_compensation.py
"""
from pathlib import Path

import torch

# train.py imports hydra at module scope, which isn't needed just to exercise
# this pure function and isn't installed in this check environment. Extract
# and exec only _apply_action_scale_compensation's source instead of
# importing the module, same approach as check_standing_reward.py.
_TRAIN_SRC = (Path(__file__).resolve().parents[1] / "train.py").read_text()
_START = _TRAIN_SRC.index("def _apply_action_scale_compensation")
_END = _TRAIN_SRC.index("\n\n@contextlib.contextmanager")
_ns = {"torch": torch}
exec(compile(_TRAIN_SRC[_START:_END], "train.py:_apply_action_scale_compensation", "exec"), _ns)
_apply_action_scale_compensation = _ns["_apply_action_scale_compensation"]


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise SystemExit(1)


ACTION_DIM = 19
OLD_SCALE_DEG, NEW_SCALE_DEG = 22.5, 90.0
old_scale = torch.full((ACTION_DIM,), OLD_SCALE_DEG * 3.141592653589793 / 180.0)
new_scale = torch.full((ACTION_DIM,), NEW_SCALE_DEG * 3.141592653589793 / 180.0)
expected_ratio = OLD_SCALE_DEG / NEW_SCALE_DEG  # 0.25


class FakeActor:
    def __init__(self, in_dim, action_dim, seed=0):
        torch.manual_seed(seed)
        self.last = torch.nn.Linear(in_dim, 2 * action_dim)


class FakeAgent:
    def __init__(self, actor):
        self.actor = actor


in_dim = 32
agent = FakeAgent(FakeActor(in_dim, ACTION_DIM))
# Snapshot pre-tanh means the OLD actor would have produced on a batch of
# typical inputs, matching the project's logged operating range (small std).
torch.manual_seed(1)
batch = torch.randn(256, in_dim) * 0.3  # small activations -> small pre-tanh mean
with torch.no_grad():
    old_out = agent.actor.last(batch)
old_mean, old_std = torch.chunk(old_out, 2, dim=-1)
old_physical_command = old_scale * torch.tanh(old_mean)

checkpoint_contract = {"control": {"action_scale_rad": old_scale.tolist()}}
current_contract = {"control": {"action_scale_rad": new_scale.tolist()}}
ratio = _apply_action_scale_compensation(agent, checkpoint_contract, current_contract)

check("returned ratio matches old_scale/new_scale", torch.allclose(ratio, torch.full((ACTION_DIM,), expected_ratio)))

with torch.no_grad():
    new_out = agent.actor.last(batch)
new_mean, new_std = torch.chunk(new_out, 2, dim=-1)
new_physical_command = new_scale * torch.tanh(new_mean)

check("std half is untouched", torch.allclose(old_std, new_std))

max_abs_error = (new_physical_command - old_physical_command).abs().max().item()
mean_abs_error = (new_physical_command - old_physical_command).abs().mean().item()
mean_abs_command = old_physical_command.abs().mean().item()

# The honest comparison is against doing nothing: applying the new scale to
# the SAME uncompensated mean, which is what a plain init_from does today.
uncompensated_physical = new_scale * torch.tanh(old_mean)
uncompensated_max_error = (uncompensated_physical - old_physical_command).abs().max().item()
uncompensated_mean_error = (uncompensated_physical - old_physical_command).abs().mean().item()

print(f"with compensation:    max_err={max_abs_error:.4f} rad  mean_err={mean_abs_error:.4f} rad "
      f"(mean |old_physical|={mean_abs_command:.4f} rad)")
print(f"without compensation: max_err={uncompensated_max_error:.4f} rad  "
      f"mean_err={uncompensated_mean_error:.4f} rad")
check("compensation cuts mean error by at least 5x vs doing nothing",
      mean_abs_error * 5 < uncompensated_mean_error)
check("compensation cuts max error by at least 3x vs doing nothing",
      max_abs_error * 3 < uncompensated_max_error)

# Demonstrate the known limitation: large pre-tanh means saturate tanh
# differently before vs after rescaling, so compensation is NOT exact there.
torch.manual_seed(2)
large_batch = torch.randn(256, in_dim) * 5.0
agent_large = FakeAgent(FakeActor(in_dim, ACTION_DIM, seed=0))
with torch.no_grad():
    old_large_out = agent_large.actor.last(large_batch)
old_large_mean, _ = torch.chunk(old_large_out, 2, dim=-1)
old_large_physical = old_scale * torch.tanh(old_large_mean)
_apply_action_scale_compensation(agent_large, checkpoint_contract, current_contract)
with torch.no_grad():
    new_large_out = agent_large.actor.last(large_batch)
new_large_mean, _ = torch.chunk(new_large_out, 2, dim=-1)
new_large_physical = new_scale * torch.tanh(new_large_mean)
large_error = (new_large_physical - old_large_physical).abs().max().item()
print(f"large-activation max |new_physical - old_physical| = {large_error:.6f} rad "
      f"(expected to be larger than the small-activation case -- tanh saturation, not a bug)")
check("large-activation error is larger than small-activation error (documents the known limitation)",
      large_error > max_abs_error)

print("\nAll action-scale compensation checks passed.")
