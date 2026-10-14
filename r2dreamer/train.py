import atexit
import contextlib
import pathlib
import signal
import sys
import warnings

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer
from training_contract import build_training_contract, check_checkpoint_contract
from script_free_contract import validate_student_config, check_imitation_source

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def _transfer_world_model(agent, checkpoint_state):
    """Load reusable perception/dynamics without importing a failed gait policy.

    The reward, actor and value functions encode the old sliding loophole and
    must start fresh after the task definition changes. The observation/action
    tensor shapes are still required to match exactly.
    """
    loaded = []
    for name in ("encoder", "rssm", "cont", "prj"):
        module = getattr(agent, name, None)
        if module is None:
            continue
        prefix = name + "."
        state = {
            key[len(prefix):]: value
            for key, value in checkpoint_state.items()
            if key.startswith(prefix)
        }
        if not state:
            raise ValueError(f"Transfer checkpoint has no {name!r} state")
        module.load_state_dict(state, strict=True)
        loaded.append(name)
    agent.clone_and_freeze()
    return tuple(loaded)


def _apply_action_scale_compensation(agent, checkpoint_contract, current_contract):
    """Rescale the actor's mean head so init_from preserves physical commands
    across an action_scale change, instead of a silent N x command-gain jump.

    bounded_normal (distributions.py) splits MLPHead.last's output in half:
    the first half parameterizes the pre-tanh Normal mean (the policy's
    mode/mean command), the second half its std. Only the mean half is
    rescaled here -- std controls exploration noise in pre-tanh space, a
    separate concern from the carried-over mean command, and left alone.

    This is an approximation, not an exact guarantee: tanh is linear only
    near zero. It holds well while the pre-tanh mean stays small (consistent
    with this project's logged action_std, typically ~0.2-0.3) and degrades
    for large pre-tanh values where tanh saturates.
    """
    old_scale = torch.as_tensor(checkpoint_contract["control"]["action_scale_rad"], dtype=torch.float32)
    new_scale = torch.as_tensor(current_contract["control"]["action_scale_rad"], dtype=torch.float32)
    if old_scale.shape != new_scale.shape:
        raise ValueError("action_scale_rad shape mismatch between checkpoint and current contract")
    if not torch.all(new_scale > 0):
        raise ValueError("action_scale_rad must be positive to compute a compensation ratio")
    action_dim = old_scale.shape[0]
    last = agent.actor.last
    if last.out_features != 2 * action_dim:
        raise ValueError(
            f"Actor head has {last.out_features} outputs, expected {2 * action_dim} "
            f"(mean+std for {action_dim} actions)")
    ratio = (old_scale / new_scale).to(device=last.weight.device, dtype=last.weight.dtype)
    with torch.no_grad():
        last.weight[:action_dim, :].mul_(ratio.unsqueeze(-1))
        last.bias[:action_dim].mul_(ratio)
    return ratio


@contextlib.contextmanager
def _graceful_stop_signals(trainer):
    """Defer OS stop requests until the trainer finishes its current update.

    No I/O or model operations run in the handler. SIGKILL, process crashes and
    device failures still rely on the last successfully saved checkpoint.
    """
    previous = {}

    def request(signum, frame):
        trainer.request_stop(signal.Signals(signum).name)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, request)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    logger = tools.Logger(logdir)
    # save config
    logger.log_hydra_config(config)

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)
    training_contract = build_training_contract(train_envs, config.env)
    if bool(getattr(config.env, "mjx_full_command", False)):
        validate_student_config(config)
    model_config = OmegaConf.to_container(config.model, resolve=True)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)
    agent.export_model_config = model_config

    # Resume support: `resume_from` (optional config field, e.g.
    # `resume_from=/path/to/step_0100000.pt`) loads a prior run's agent + optimizer
    # weights and carries its absolute step count forward, so eval/checkpoint/log
    # cadence and the `steps` budget stay continuous instead of restarting at 0.
    # NOT restored: the replay buffer's contents (checkpoints never saved them) --
    # a resumed run starts with an empty buffer and briefly refills before updates
    # resume, the same cost as the normal pretraining warm-up.
    step_offset = 0
    update_offset = 0
    imitation_dataset = None
    imitation_metadata = None
    if bool(getattr(config.env, "mjx_full_command", False)):
        from imitation_data import ImitationDataset
        if not config.imitation.dataset:
            raise ValueError("Full-command training requires the fingerprinted expert dataset")
        imitation_dataset = ImitationDataset(config.imitation.dataset)
        check_imitation_source(imitation_dataset, training_contract)
    resume_from = getattr(config, "resume_from", None)
    init_from = getattr(config, "init_from", None)
    transfer_from = getattr(config, "transfer_from", None)
    transfer_additional_steps = int(getattr(config, "transfer_additional_steps", 0))
    resume_reset_optimizer = bool(getattr(config, "resume_reset_optimizer", False))
    if sum(bool(path) for path in (resume_from, init_from, transfer_from)) > 1:
        raise ValueError(
            "Choose exactly one of resume_from, init_from or transfer_from")
    if resume_reset_optimizer and not resume_from:
        raise ValueError("resume_reset_optimizer requires resume_from")
    if transfer_additional_steps and not transfer_from:
        raise ValueError("transfer_additional_steps requires transfer_from")
    if init_from:
        # A new curriculum stage starts a fresh budget and optimizer, retaining
        # learned weights. Observation/action contracts must still match exactly.
        ckpt = torch.load(pathlib.Path(init_from).expanduser(), map_location=config.device)
        if imitation_dataset is not None:
            imitation_metadata = check_imitation_source(
                imitation_dataset, training_contract, ckpt,
                minimum_updates=int(config.imitation.pretrain_updates))
        for message in check_checkpoint_contract(ckpt, training_contract, mode="init"):
            print(message)
        agent.load_state_dict(ckpt["agent_state_dict"])
        if bool(getattr(config, "init_from_action_scale_compensation", False)):
            saved_contract = ckpt.get("training_contract")
            if not saved_contract:
                raise ValueError(
                    "init_from_action_scale_compensation requires a checkpoint with "
                    "training_contract metadata")
            ratio = _apply_action_scale_compensation(agent, saved_contract, training_contract)
            print(f"Action-scale compensation applied to actor mean head: ratio={ratio.tolist()}")
        if bool(getattr(config.env, "encoder_training_aid", False)):
            if "optimizer_updates" not in ckpt:
                raise ValueError("Encoder-aid transfer requires a checkpoint with optimizer_updates")
            update_offset = int(ckpt["optimizer_updates"])
        print(f"Initialized new stage from {init_from}; fresh optimizer and replay")
    if transfer_from:
        transfer_path = pathlib.Path(transfer_from).expanduser()
        ckpt = torch.load(transfer_path, map_location=config.device)
        for message in check_checkpoint_contract(ckpt, training_contract, mode="transfer"):
            print(message)
        loaded = _transfer_world_model(agent, ckpt["agent_state_dict"])
        if "optimizer_updates" not in ckpt:
            raise ValueError("World-model transfer requires optimizer_updates metadata")
        update_offset = int(ckpt["optimizer_updates"])
        step_offset = int(ckpt.get("step", 0))
        if transfer_additional_steps <= 0:
            raise ValueError("transfer_additional_steps must be positive")
        absolute_target = step_offset + transfer_additional_steps
        config.trainer.steps = absolute_target
        print(
            f"Transferred {', '.join(loaded)} from {transfer_path}; reset actor, "
            f"reward/value heads, optimizer and replay. Training {transfer_additional_steps} "
            f"additional steps ({step_offset} -> {absolute_target})."
        )
    if resume_from:
        resume_path = pathlib.Path(resume_from).expanduser()
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=config.device)
        if imitation_dataset is not None:
            imitation_metadata = check_imitation_source(
                imitation_dataset, training_contract, ckpt,
                minimum_updates=int(config.imitation.pretrain_updates))
        for message in check_checkpoint_contract(
                ckpt, training_contract, mode="resume",
                allow_unverified_resume=bool(getattr(config, "allow_unverified_resume", False))):
            print(message)
        agent.load_state_dict(ckpt["agent_state_dict"])
        if resume_reset_optimizer:
            print("Using a fresh optimizer and learning-rate schedule for recovery")
        else:
            tools.recursively_load_optim_state_dict(agent, ckpt["optims_state_dict"])
        step_offset = int(ckpt.get("step", 0))
        if bool(getattr(config.env, "encoder_training_aid", False)) and "optimizer_updates" not in ckpt:
            raise ValueError("Encoder-aid resume requires a checkpoint with optimizer_updates")
        update_offset = int(ckpt.get("optimizer_updates", 0))
        if not resume_reset_optimizer:
            # Older checkpoints saved Adam state but omitted the LambdaLR state.
            # Restore its phase so a mature policy does not repeat warmup on
            # every resume (which previously reduced LR almost to zero).
            scheduler = agent._scheduler
            scheduler.base_lrs = [group.get("initial_lr", base)
                                 for group, base in zip(
                                     agent._optimizer.param_groups, scheduler.base_lrs)]
            scheduler.last_epoch = update_offset
            scheduler._step_count = update_offset + 1
            scheduler._last_lr = [base * schedule(update_offset)
                                  for base, schedule in zip(
                                      scheduler.base_lrs, scheduler.lr_lambdas)]
            for group, lr in zip(agent._optimizer.param_groups, scheduler._last_lr):
                group["lr"] = lr
            print(f"Restored learning-rate schedule at update {update_offset}: "
                  f"lr={scheduler.get_last_lr()}")
        print(f"Resumed at step {step_offset}")

    if imitation_dataset is not None and not (init_from or resume_from):
        raise ValueError("Full-command online training must start from its verified imitation checkpoint")
    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)
    policy_trainer.training_contract = training_contract
    policy_trainer.imitation_dataset = imitation_dataset
    policy_trainer.imitation_settings = getattr(config, "imitation", None)
    policy_trainer.imitation_metadata = imitation_metadata
    if imitation_dataset is not None:
        policy_trainer.imitation_rng = np.random.default_rng(int(config.imitation.seed))
    if resume_from:
        policy_trainer.restore_checkpoint_selection(ckpt)
    with _graceful_stop_signals(policy_trainer):
        policy_trainer.begin(agent, step_offset=step_offset, update_offset=update_offset)
        policy_trainer.save_checkpoint(agent, "latest.pt", step=policy_trainer.last_train_step)


if __name__ == "__main__":
    main()
