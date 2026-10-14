"""Episode-separated, causally aligned demonstrations for a script-free actor."""
from __future__ import annotations

import hashlib
import os
import json
from numbers import Integral
from pathlib import Path

import numpy as np

SPLITS = {"train": 0, "validation": 1, "val": 1, "test": 2}
REQUIRED = {"proprio", "action", "reward", "is_terminal", "valid", "split"}


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_arrays(arrays):
    if not REQUIRED <= arrays.keys():
        raise ValueError(f"Missing demonstration arrays: {sorted(REQUIRED - arrays.keys())}")
    proprio, action = np.asarray(arrays["proprio"]), np.asarray(arrays["action"])
    if action.ndim != 3 or action.shape[2] != 19 or min(action.shape[:2]) < 1:
        raise ValueError("Actions must have shape (episodes, transitions, 19)")
    episodes, steps, _ = action.shape
    if proprio.shape != (episodes, steps + 1, 45):
        raise ValueError("Proprioception must include obs_t through obs_T with shape (E,T+1,45)")
    for name in ("reward", "is_terminal", "valid"):
        if np.asarray(arrays[name]).shape != (episodes, steps):
            raise ValueError(f"Invalid demonstration shape: {name}")
    split = np.asarray(arrays["split"])
    if split.shape != (episodes,) or split.dtype.kind not in "iu" or not np.isin(split, [0, 1, 2]).all():
        raise ValueError("Splits must be episode IDs 0=train, 1=validation, 2=test")
    present = np.asarray(arrays.get("present", np.ones((episodes, steps), dtype=bool)), dtype=bool)
    valid = np.asarray(arrays["valid"], dtype=bool)
    if present.shape != valid.shape or np.any(valid & ~present):
        raise ValueError("Safe expert labels cannot exist in padded transitions")
    if np.any(np.diff(present.astype(np.int8), axis=1) > 0):
        raise ValueError("Each episode must have a contiguous observed prefix")
    for name, array in arrays.items():
        value = np.asarray(array)
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"Object, nonnumeric or nonfinite demonstration data: {name}")
    if np.any(np.abs(action) > 1.):
        raise ValueError("Expert labels must be normalized full motor commands in [-1,1]")
    for name in ("valid", "is_terminal", "present"):
        if name in arrays and not np.isin(arrays[name], [False, True]).all():
            raise ValueError(f"Demonstration mask must be binary: {name}")
    terminal = np.asarray(arrays["is_terminal"], dtype=bool)
    last_observed = present.sum(axis=1) - 1
    if np.any(terminal & (~present | (np.arange(steps)[None] != last_observed[:, None]))):
        raise ValueError("Terminal flags must occur only on an episode's final observed transition")
    if "episode_seed" in arrays:
        seeds = np.asarray(arrays["episode_seed"])
        if seeds.shape != (episodes,):
            raise ValueError("episode_seed must identify each rollout")
        for seed in np.unique(seeds):
            if len(np.unique(split[seeds == seed])) != 1:
                raise ValueError("A rollout seed cannot leak across train/validation/test")
    if "rollout_id" in arrays and len(np.unique(arrays["rollout_id"])) != episodes:
        raise ValueError("Rollout IDs must be unique")
    return dict(episodes=episodes, transitions_per_episode=steps,
                present_transitions=int(present.sum()), safe_expert_labels=int(valid.sum()),
                split_counts={name: int((split == code).sum()) for name, code in SPLITS.items() if name != "val"})


def split_statistics(arrays):
    present = np.asarray(arrays.get("present", np.ones_like(arrays["valid"], dtype=bool)), dtype=bool)
    result = {}
    for name, code in (("train", 0), ("validation", 1), ("test", 2)):
        selected = np.asarray(arrays["split"]) == code
        result[name] = dict(episodes=int(selected.sum()), observed_transitions=int(present[selected].sum()),
                            safe_expert_labels=int(np.asarray(arrays["valid"])[selected].sum()),
                            safe_labels_after_64_frames=int(np.asarray(arrays["valid"])[selected, 64:].sum()),
                            seeds=sorted(set(np.asarray(arrays.get("episode_seed", np.arange(len(selected))))[selected].tolist())))
    return result


def align_episode(arrays, episode, start=0, length=None, burnin=0):
    """obs_t receives a_(t-1)/r_(t-1); current a_t is a supervised target only."""
    total = arrays["action"].shape[1]
    indices = (episode, start, burnin) if length is None else (episode, start, burnin, length)
    if any(not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) for value in indices):
        raise ValueError("Episode/window/burn-in indices must be integers")
    length = total + 1 - start if length is None else length
    if min(start, burnin) < 0 or length < 1 or burnin >= length or not 0 <= episode < len(arrays["split"]):
        raise ValueError("Invalid episode/window/burn-in request")
    present = np.asarray(arrays.get("present", np.ones_like(arrays["valid"], dtype=bool))[episode], dtype=bool)
    observed = int(present.sum())
    if start > observed:
        raise ValueError("A demonstration window must start within its observed episode")
    times = np.arange(start, start + length)
    in_range = (times >= 0) & (times <= observed)
    result = dict(proprio=np.zeros((length, 45), np.float32),
                  action=np.zeros((length, 19), np.float32), reward=np.zeros((length, 1), np.float32),
                  is_terminal=np.zeros((length, 1), bool), is_first=np.zeros((length, 1), bool),
                  expert_action=np.zeros((length, 19), np.float32),
                  expert_mask=np.zeros((length, 1), np.float32), loss_mask=np.zeros((length, 1), np.float32))
    for offset, time in enumerate(times):
        if not in_range[offset]:
            continue
        result["proprio"][offset] = arrays["proprio"][episode, time]
        result["is_first"][offset, 0] = time == 0
        result["loss_mask"][offset, 0] = offset >= burnin
        if time > 0:
            # DART data: the encoder sees the command actually executed (noisy); the label stays clean.
            executed = arrays["executed_action"] if "executed_action" in arrays else arrays["action"]
            result["action"][offset] = executed[episode, time - 1]
            result["reward"][offset, 0] = arrays["reward"][episode, time - 1]
            result["is_terminal"][offset, 0] = arrays["is_terminal"][episode, time - 1]
        if time < observed:
            result["expert_action"][offset] = arrays["action"][episode, time]
            result["expert_mask"][offset, 0] = bool(arrays["valid"][episode, time]) and offset >= burnin
    return result


def save_dataset(path, arrays, manifest):
    """Write a new non-pickle NPZ and evidence manifest, preserving existing data."""
    path = Path(path).resolve()
    manifest_path = path.with_suffix(".manifest.json")
    if path.exists() or manifest_path.exists():
        raise FileExistsError("Demonstration dataset or manifest exists; choose a fresh output")
    summary = validate_arrays(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    result = dict(manifest, schema_version=1, data_file=path.name, data_sha256=file_sha256(path),
                  arrays={name: dict(shape=list(np.asarray(value).shape), dtype=str(np.asarray(value).dtype))
                          for name, value in arrays.items()}, summary=summary,
                  alignment="obs_t + previous action/reward/terminal; current expert_action is a label, never an encoder input",
                  policy_observations=["proprio:45"], split_definition="whole rollout seed groups, never individual frames",
                  split_statistics=split_statistics(arrays))
    manifest_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


class ImitationDataset:
    def __init__(self, path, manifest_path=None):
        self.path = Path(path).resolve(strict=True)
        manifest_path = (self.path.with_suffix(".manifest.json") if manifest_path is None else Path(manifest_path)).resolve(strict=True)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.fingerprint = dict(data_sha256=file_sha256(self.path), manifest_sha256=file_sha256(manifest_path))
        if self.manifest.get("schema_version") != 1 or self.manifest.get("data_sha256") != self.fingerprint["data_sha256"]:
            raise ValueError("Demonstration schema/hash does not match its manifest")
        with np.load(self.path, allow_pickle=False) as source:
            self.arrays = {name: source[name].copy() for name in source.files}
        self.summary = validate_arrays(self.arrays)
        self.split_statistics = split_statistics(self.arrays)
        self.rng = np.random.default_rng(0)

    def numpy_batch(self, batch_size, batch_length, burnin=64, split="train", rng=None):
        if min(batch_size, batch_length) < 1 or burnin < 0:
            raise ValueError("Batch sizes must be positive and burn-in nonnegative")
        code = SPLITS[split] if isinstance(split, str) else int(split)
        generator = self.rng if rng is None else np.random.default_rng(rng) if isinstance(rng, (int, np.integer)) else rng
        length = burnin + batch_length
        candidates, startup_candidates = [], []
        for episode in np.flatnonzero(self.arrays["split"] == code):
            safe_times = np.flatnonzero(self.arrays["valid"][episode])
            if np.any(safe_times < length):
                startup_candidates.append(int(episode))
            safe_times = safe_times[safe_times >= burnin]
            if len(safe_times):
                candidates.append((int(episode), safe_times))
        if not startup_candidates and not candidates:
            raise ValueError(f"No safe expert labels in split {split!r}; collect qualified demonstrations before behavior cloning")
        windows = []
        startup_count = max(1, int(np.ceil(batch_size * float(os.environ.get("STARTUP_FRACTION", .25))))) if startup_candidates else 0
        for index in range(batch_size):
            if index < startup_count or not candidates:
                episode = int(generator.choice(startup_candidates))
                # The true rollout beginning has a known zero recurrent state:
                # supervise initiation immediately, with no artificial burn-in.
                windows.append(align_episode(self.arrays, episode, 0, length, burnin=0))
                continue
            episode, safe_times = candidates[int(generator.integers(len(candidates)))]
            time = int(generator.choice(safe_times))
            first = max(0, time - length + 1)
            last = time - burnin
            start = int(generator.integers(first, last + 1))
            windows.append(align_episode(self.arrays, episode, start, length, burnin))
        result = {key: np.stack([window[key] for window in windows]) for key in windows[0]}
        if np.any(result["expert_mask"].sum(axis=(1, 2)) == 0):
            raise RuntimeError("Sampler produced an empty expert-label window")
        return result

    def sample(self, batch_size, batch_length, burnin=64, split="train", device="cpu", rng=None):
        import torch
        from tensordict import TensorDict

        values = self.numpy_batch(batch_size, batch_length, burnin, split, rng)
        values = {key: torch.as_tensor(value, device=device) for key, value in values.items()}
        return TensorDict(values, batch_size=(batch_size, burnin + batch_length), device=device)

    batch = sample
