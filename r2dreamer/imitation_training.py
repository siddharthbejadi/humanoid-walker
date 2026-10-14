"""Observed-state full-command supervision; expert labels never enter the encoder."""
from pathlib import Path

import torch
from learning_diagnostics import barlow_objective


def sequence_mask(data, name):
    value = data[name]
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.shape != data["proprio"].shape[:2] or not torch.isfinite(value).all():
        raise ValueError(f"Invalid {name} shape or values")
    if not ((value == 0) | (value == 1)).all():
        raise ValueError(f"{name} must be binary")
    return value.bool()


def masked_mean(value, mask):
    if not mask.any():
        raise ValueError("No valid post-burn-in samples for this loss")
    if value.shape[:2] != mask.shape:
        raise ValueError("Loss and sequence mask shape mismatch")
    while value.ndim > 2:
        value = value.mean(-1)
    if not torch.isfinite(value[mask]).all():
        raise ValueError("Nonfinite supervised/world-model loss")
    return value[mask].mean()


def imitation_objective(agent, data, initial, include_world=True, deterministic=False):
    labels = data["expert_action"]
    if labels.shape != (*data["proprio"].shape[:2], agent.act_dim):
        raise ValueError("Expert command shape mismatch")
    if not torch.isfinite(labels).all() or (labels.abs() > 1).any():
        raise ValueError("Expert commands must be finite normalized full commands")
    world_mask = sequence_mask(data, "loss_mask")
    expert_mask = sequence_mask(data, "expert_mask") & world_mask
    if not expert_mask.any():
        raise ValueError("Expert batch has no valid post-burn-in labels")
    if not torch.isfinite(data["proprio"]).all() or not torch.isfinite(data["action"]).all():
        raise ValueError("Nonfinite expert observation/previous action")
    embed = agent.encoder(data)
    if deterministic:
        # Match deployment's posterior modes while retaining causal expert context.
        stoch, deter = initial
        states, hidden, logits = [], [], []
        for time in range(embed.shape[1]):
            stoch, deter, posterior = agent.rssm.obs_step(stoch, deter, data["action"][:, time],
                embed[:, time], data["is_first"][:, time], sample=False)
            states.append(stoch)
            hidden.append(deter)
            logits.append(posterior)
        stoch, deter, posterior = (torch.stack(values, dim=1) for values in (states, hidden, logits))
    else:
        stoch, deter, posterior = agent.rssm.observe(embed, data["action"], initial, data["is_first"])
    feat = agent.rssm.get_feat(stoch, deter)
    mean = agent.actor(feat).mode
    bc = masked_mean((mean - labels).square(), expert_mask)
    losses = {"bc": bc}
    if include_world:
        _, prior = agent.rssm.prior(deter)
        dyn, rep = agent.rssm.kl_loss(posterior, prior, agent.kl_free)
        losses.update(dyn=masked_mean(dyn, world_mask), rep=masked_mean(rep, world_mask),
            rew=masked_mean(-agent.reward(feat).log_prob(data["reward"].float()), world_mask),
            con=masked_mean(-agent.cont(feat).log_prob(1. - data["is_terminal"].float()), world_mask))
        if agent.rep_loss != "r2dreamer":
            raise ValueError("Offline imitation pretraining currently requires rep_loss=r2dreamer")
        selected = world_mask.clone()
        selected[:, torch.arange(selected.shape[1], device=selected.device) % agent._barlow_stride != 0] = False
        if selected.sum() < 2:
            selected = world_mask
        if selected.sum() < 2:
            raise ValueError("Representation pretraining needs at least two valid frames")
        projected = agent.prj(feat[selected]).unsqueeze(1)
        target = embed[selected].detach().unsqueeze(1)
        losses["barlow"], _ = barlow_objective(projected, target, agent.barlow_lambd,
                                              agent._barlow_reduction, 1)
    weighted = sum(loss * (agent.imitation_pretrain_weight if name == "bc" else
                           agent._loss_scales[name]) for name, loss in losses.items())
    metrics = {"imitation/" + name: value.detach() for name, value in losses.items()}
    metrics["imitation/labels"] = expert_mask.sum().detach()
    metrics["imitation/total_loss"] = weighted.detach()
    return weighted, metrics


@torch.no_grad()
def full_episode_imitation_evidence(agent, dataset, split):
    """Score recorded episodes with true reset/history, not counterfactual physics."""
    import numpy as np
    from tensordict import TensorDict
    from imitation_data import SPLITS, align_episode

    if split not in ("validation", "test"):
        raise ValueError("Full-episode evidence requires a held-out split")
    arrays = dataset.arrays
    episodes = np.flatnonzero(arrays["split"] == SPLITS[split])
    if not len(episodes):
        raise ValueError(f"No held-out episodes in split {split!r}")
    present = np.asarray(arrays.get("present", np.ones_like(arrays["valid"], dtype=bool)), dtype=bool)
    records, total_error, total_labels = [], 0., 0
    was_training = agent.training
    agent.eval()
    try:
        for episode in episodes:
            observed = int(present[episode].sum())
            labels = int(np.asarray(arrays["valid"][episode, :observed], dtype=bool).sum())
            record = {"episode": int(episode), "observed_control_transitions": observed,
                      "labels": labels, "bc_mse": None,
                      "exclusion_reason": "no_safe_expert_labels" if not labels else None}
            for name, target in (("episode_seed", "seed"), ("rollout_id", "rollout_id")):
                if name in arrays:
                    record[target] = int(arrays[name][episode])
            records.append(record)
            if not labels:
                continue
            aligned = align_episode(arrays, int(episode), start=0, length=observed + 1, burnin=0)
            data = TensorDict({name: torch.as_tensor(value[None], device=agent.device)
                               for name, value in aligned.items()}, batch_size=(1, observed + 1))
            target = data["expert_action"][0]
            if not torch.isfinite(target).all() or (target.abs() > 1).any():
                raise ValueError("Held-out commands must be finite normalized full commands")
            embed = agent.encoder(agent.preprocess(data))
            stoch, deter = agent.rssm.initial(1)
            predictions = []
            # Carry the actual episode history rather than restarting each window.
            for time in range(observed + 1):
                stoch, deter, _ = agent.rssm.obs_step(stoch, deter, data["action"][:, time],
                    embed[:, time], data["is_first"][:, time], sample=False)
                predictions.append(agent.actor(agent.rssm.get_feat(stoch, deter)).mode[0])
            prediction = torch.stack(predictions)
            if not torch.isfinite(prediction).all():
                raise ValueError("Nonfinite full-episode held-out commands")
            mask = sequence_mask(data, "expert_mask")[0] & sequence_mask(data, "loss_mask")[0]
            error = (prediction - target).square().mean(-1)[mask]
            if error.numel() != labels or not torch.isfinite(error).all():
                raise ValueError("Held-out label mask or command error is invalid")
            squared_error = float(error.sum())
            total_error += squared_error
            total_labels += labels
            record["bc_mse"] = squared_error / labels
            del data, embed, prediction, predictions, target, stoch, deter
    finally:
        agent.train(was_training)
    if not total_labels:
        raise ValueError(f"No safe expert labels for full-episode evidence in split {split!r}")
    return {"protocol": "offline_teacher_forced_full_episode_v1", "split": split,
            "labels": total_labels, "bc_mse": total_error / total_labels,
            "episode_count": len(records), "scored_episode_count": sum(record["labels"] > 0 for record in records),
            "observed_control_transitions": sum(record["observed_control_transitions"] for record in records),
            "per_episode": records, "deterministic_posterior": True,
            "previous_action_source": "recorded_expert_commands",
            "observation_source": "recorded_expert_observations", "walking_qualified": False,
            "scope": "Offline teacher-forced command prediction on complete recorded episodes; not actual closed-loop physics or walking"}


def save_imitation_checkpoint(agent, path, contract, metadata, updates):
    import tools
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
                "step": 0, "optimizer_updates": int(updates), "stop_reason": None,
                "training_contract": contract, "imitation": metadata,
                "model_config": getattr(agent, "export_model_config", None)}, temporary)
    temporary.replace(path)
