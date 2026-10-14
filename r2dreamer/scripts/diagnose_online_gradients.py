"""Compare saved students on fixed data without optimization or simulation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from imitation_data import ImitationDataset, file_sha256
from imitation_training import imitation_objective
from script_free_policy import ScriptFreePolicy


def compare_gradients(first, second):
    """Treat absent gradients as zero when comparing the same parameter group."""
    keys = first.keys() | second.keys()
    first_sq = second_sq = dot = 0.
    for name in keys:
        left, right = first.get(name), second.get(name)
        if left is not None:
            first_sq += float(left.double().square().sum())
        if right is not None:
            second_sq += float(right.double().square().sum())
        if left is not None and right is not None:
            if left.shape != right.shape:
                raise ValueError(f"Gradient shape differs for {name}")
            dot += float((left.double() * right.double()).sum())
    first_norm, second_norm = first_sq ** .5, second_sq ** .5
    return {"online_norm": first_norm, "bc_anchor_norm": second_norm,
            "online_to_bc_norm_ratio": first_norm / second_norm if second_norm else None,
            "cosine": dot / (first_norm * second_norm) if first_norm and second_norm else None}


def gradient_groups(agent):
    result = {"actor": {}, "encoder_rssm": {}}
    for prefix, module, group in (("actor", agent.actor, "actor"),
                                  ("encoder", agent.encoder, "encoder_rssm"),
                                  ("rssm", agent.rssm, "encoder_rssm")):
        for name, parameter in module.named_parameters():
            if parameter.grad is not None:
                value = parameter.grad.detach().float().cpu().clone()
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite gradient in {prefix}.{name}")
                result[group][prefix + "." + name] = value
    return result


def parameter_digest(agent):
    digest = hashlib.sha256()
    for name, value in agent.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fixed_batch(dataset, batch_size, length, burnin, split, seed):
    generator = np.random.default_rng(seed)
    for _ in range(64):
        values = dataset.numpy_batch(batch_size, length, burnin, split, generator)
        tail = slice(burnin, burnin + length)
        if (np.all(values["loss_mask"][:, tail] == 1) and
                np.all(values["expert_mask"][:, tail] == 1) and
                not np.any(values["is_terminal"][:, tail])):
            return values
    raise ValueError(f"Could not sample a complete safe {split} window")


def on_device(values, device):
    shape = values["proprio"].shape[:2]
    return TensorDict({key: torch.as_tensor(value, device=device)
                       for key, value in values.items()}, batch_size=shape, device=device)


def diagnose(checkpoint_path, dataset, training_values, validation_values, device,
             burnin, seed, anchor_weight):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint = deepcopy(checkpoint)
    checkpoint["model_config"]["compile"] = False
    checkpoint["model_config"]["mixed_precision"] = False
    agent = ScriptFreePolicy(checkpoint, device=device).agent
    fingerprint = checkpoint["imitation"].get("dataset_fingerprint")
    if fingerprint != dataset.fingerprint:
        raise ValueError("Checkpoint and demonstration fingerprints differ")
    before = parameter_digest(agent)
    training = on_device(training_values, device)
    validation = on_device(validation_values, device)
    validation_metrics = agent.validate_imitation(validation)
    agent._diagnostic_due = False
    agent._optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    # The prefix reconstructs recurrent context; only the complete tail is used
    # for online losses, so padded or artificial zero-context windows are absent.
    initial = agent.rssm.initial(training.batch_size[0])
    if burnin:
        with torch.no_grad():
            embed = agent.encoder(training[:, :burnin])
            stoch, deter, _ = agent.rssm.observe(embed, training["action"][:, :burnin],
                                                initial, training["is_first"][:, :burnin])
            initial = (stoch[:, -1].detach(), deter[:, -1].detach())
    tail = training[:, burnin:].clone()
    tail.set("is_last", tail["is_terminal"].clone())
    ema_before = agent.return_ema.ema_vals.clone()
    _, online_metrics = agent._cal_grad(agent.preprocess(tail), initial)
    online_gradients = gradient_groups(agent)
    agent._optimizer.zero_grad(set_to_none=True)
    agent.return_ema.ema_vals.copy_(ema_before)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    bc_loss, bc_metrics = imitation_objective(
        agent, agent.preprocess(training), agent.rssm.initial(training.batch_size[0]),
        include_world=False)
    anchor = anchor_weight * bc_loss / agent.imitation_pretrain_weight
    anchor.backward()
    bc_gradients = gradient_groups(agent)
    agent._optimizer.zero_grad(set_to_none=True)
    after = parameter_digest(agent)
    if before != after:
        raise RuntimeError("Diagnostic changed model parameters")
    metrics = {name: float(value.detach().float().cpu())
               for name, value in online_metrics.items()
               if name in ("loss/policy", "loss/value", "loss/repval", "adv", "adv_std",
                           "action_entropy", "rew", "val", "tar", "con")}
    result = {"checkpoint": str(checkpoint_path), "checkpoint_step": int(checkpoint.get("step", 0)),
              "optimizer_updates": int(checkpoint.get("optimizer_updates", 0)),
              "validation_bc_mse": float(validation_metrics["imitation/bc"]),
              "sampled_training_bc_mse": float(bc_metrics["imitation/bc"]),
              "anchor_weight": anchor_weight,
              "gradients": {group: compare_gradients(online_gradients[group], bc_gradients[group])
                            for group in online_gradients},
              "online_metrics": metrics, "parameters_before_sha256": before,
              "parameters_after_sha256": after, "parameters_unchanged": True,
              "imagination_horizon": agent.imag_horizon}
    del agent, checkpoint, training, validation, online_gradients, bc_gradients
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="Repeat NAME=PATH for imitation, best and latest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--burnin", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batch-length", type=int, default=8)
    parser.add_argument("--anchor-weight", type=float, default=.1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh diagnostic report path")
    if min(args.batch_size, args.batch_length) < 1 or args.burnin < 0 or args.anchor_weight < 0:
        raise ValueError("Invalid diagnostic batch or anchor settings")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoints = {}
    for item in args.checkpoint:
        name, separator, path = item.partition("=")
        if not separator or not name or name in checkpoints:
            raise ValueError("Checkpoints must have distinct NAME=PATH entries")
        checkpoints[name] = Path(path).resolve(strict=True)
    dataset = ImitationDataset(args.dataset)
    manifest = dataset.path.with_suffix(".manifest.json")
    inputs = [dataset.path, manifest, *checkpoints.values()]
    before = {str(path): file_sha256(path) for path in inputs}
    training = fixed_batch(dataset, args.batch_size, args.batch_length, args.burnin, "train", args.seed)
    validation = fixed_batch(dataset, 4, 16, args.burnin, "validation", args.seed + 1)
    results = {name: diagnose(path, dataset, training, validation, args.device,
                             args.burnin, args.seed, args.anchor_weight)
               for name, path in checkpoints.items()}
    after = {str(path): file_sha256(path) for path in inputs}
    if before != after:
        raise RuntimeError("A diagnostic input changed during the check")
    report = {"status": "completed_read_only_gradient_diagnostic", "seed": args.seed,
              "training_batch": {"size": args.batch_size, "length": args.batch_length,
                                 "burnin": args.burnin},
              "validation_batch": {"size": 4, "length": 16, "burnin": args.burnin},
              "dataset_fingerprint": dataset.fingerprint, "input_sha256": before,
              "inputs_unchanged": True, "optimizer_steps": 0, "simulator_steps": 0,
              "checkpoints": results,
              "scope": "Fixed teacher-forced window prediction and unclipped diagnostic gradients. "
                       "Online actor gradients arise from imagined returns; shared gradients include "
                       "world-model and value losses. This is not a causal diagnosis or a walking test."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
