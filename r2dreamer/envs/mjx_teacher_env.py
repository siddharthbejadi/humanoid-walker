"""Experimental CUDA-only MJX-Warp to Torch bridge for R2Dreamer.

Physics/reward run in JAX/MJX; policy/world-model tensors remain in Torch.
DLPack exchanges actions and observations on the same GPU. This backend must
pass the supplied live CUDA smoke before any paid long training run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from tensordict import TensorDict

from .teacher_state import TEACHER_STATE_DIM


class MJXTeacherEnv:
    def __init__(self, config, env_num=None, evaluation=False):
        self.full_command = bool(getattr(config, "mjx_full_command", False))
        if config.task != "humanoid_progressive_walking":
            raise ValueError("MJX teacher supports only humanoid_progressive_walking")
        if str(config.device) != "cuda:0" or not torch.cuda.is_available():
            raise RuntimeError("MJX teacher requires CUDA Torch on cuda:0; no CPU fallback")
        import jax
        import jax.numpy as jnp
        from mujoco import mjx
        from .mjx_walking.env import MJXWalkingEnv

        if jax.default_backend() != "gpu" or not jax.devices("gpu"):
            raise RuntimeError("MJX teacher requires a working JAX GPU; no CPU fallback")
        if not bool(getattr(config, "mjx_teacher_experimental", False)):
            raise ValueError("Explicit mjx_teacher_experimental opt-in required")
        if bool(config.enable_vision):
            raise ValueError("MJX teacher camera is not implemented; set enable_vision=false")
        scene = Path(str(config.scene_path)).expanduser().resolve(strict=True)
        # The recipe may opt evaluation into its own reproducible pose mix.
        # Resolve the bundled reference relative to r2dreamer/.
        rsi_active = bool(getattr(config, "rsi_stratified", False)) and (
            not evaluation or bool(getattr(config, "eval_pose_rsi", False)))
        probabilities = (getattr(config, "eval_rsi_probabilities", None)
                         if evaluation else None)
        if probabilities is None:
            probabilities = getattr(config, "rsi_probabilities", (.10, .42, .42, .06))
        reference = None
        scripted_pool = getattr(config, "mjx_scripted_phase_pool", None)
        if rsi_active and not scripted_pool:
            configured = (getattr(config, "pose_rsi_json", None)
                          or getattr(config, "gait_reference_csv", None))
            if configured is None:
                raise ValueError("Stratified MJX RSI requires pose_rsi_json or gait_reference_csv")
            reference = Path(str(configured)).expanduser()
            if not reference.is_absolute():
                reference = Path(__file__).resolve().parents[1] / reference
            reference = reference.resolve(strict=True)
        self.env_num = int(config.env_num if env_num is None else env_num)
        self.device = torch.device(config.device)
        self.horizon = int(config.time_limit) // int(config.action_repeat)
        if self.horizon < 1:
            raise ValueError("MJX teacher requires a positive episode horizon")
        self.jax, self.jnp = jax, jnp
        self.evaluation = bool(evaluation)
        self._balanced_evaluation = self.evaluation and bool(
            getattr(config, "eval_balanced_pose_ids", False))
        self._repeatable_evaluation = self.evaluation and bool(
            getattr(config, "eval_repeatable", False))
        configured_eval_seed = getattr(config, "eval_seed", None)
        self._evaluation_seed = int(config.seed if configured_eval_seed is None
                                    else configured_eval_seed) + 10000
        self.rng = jax.random.PRNGKey(self._evaluation_seed if evaluation else int(config.seed))

        class PrivilegedWalk(MJXWalkingEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                m = self.m
                self.act_dof = np.asarray(m.jnt_dofadr[m.actuator_trnid[:, 0]], dtype=np.int32)
                self.force_scale = jnp.asarray(
                    np.maximum(1.0, np.max(np.abs(m.actuator_forcerange), axis=1)),
                    dtype=jnp.float32)

            @property
            def observation_size(self):
                return 45 if self.full_command else 45 + TEACHER_STATE_DIM

            def _observation(self, data, last_command, filtered_fsr, step_count, dot=None, phase=None, nav=None, imu_key=None):
                regular = super()._observation(data, last_command, filtered_fsr, step_count, dot=dot, phase=phase, nav=nav, imu_key=imu_key)
                if self.full_command:
                    return regular
                fsr, load, clearance, foot_xy = self._features(data)
                actual = data.qpos[self.act_qpos]
                base_velocity = data.qvel[:6]
                com_xy = data.subtree_com[self.base_id, :2]
                privileged = jnp.concatenate((
                    jnp.clip(2.0 * (actual - self.low) / self.span - 1.0, -1.0, 1.0),
                    jnp.clip(data.qvel[self.act_dof] / 10.0, -1.0, 1.0),
                    jnp.clip((data.ctrl - actual) / self.span, -1.0, 1.0),
                    jnp.clip(data.qfrc_actuator[self.act_dof] / self.force_scale, -1.0, 1.0),
                    jnp.clip(base_velocity[:3] / 2.0, -1.0, 1.0),
                    jnp.clip(base_velocity[3:] / 5.0, -1.0, 1.0),
                    jnp.clip((com_xy - jnp.mean(foot_xy, axis=0)) / .20, -1.0, 1.0),
                    (load > .02).astype(jnp.float32),
                    jnp.clip(load, 0.0, 5.0) / 5.0,
                    jnp.clip(clearance, 0.0, .10) / .10,
                    jnp.clip(fsr, 0.0, 5.0) / 5.0,
                ))
                if self.task == "standing":
                    # Standing is explicitly a zero-speed, centered-support task.
                    regular = regular.at[34:37].set(jnp.asarray([0., 0., 1.], dtype=jnp.float32))
                return jnp.concatenate((regular, privileged))

        self.raw = PrivilegedWalk(scene, worlds=self.env_num,
            contacts_per_world=int(getattr(config, "mjx_contacts_per_world", 256)),
            action_repeat=int(config.action_repeat),
            full_command=self.full_command,
            reset_joint_noise_rad=float(getattr(config, "mjx_reset_joint_noise_rad", .005)),
            reset_velocity_noise=float(getattr(config, "mjx_reset_velocity_noise", 0.)),
            target_forward_speed=float(getattr(config, "target_vx", .08)),
            arm_action_scale=float(getattr(config, "mjx_arm_action_scale", .18)),
            leg_action_scale=float(getattr(config, "mjx_leg_action_scale", .18)),
            other_action_scale=float(getattr(config, "mjx_other_action_scale", .18)),
            control_slew_rad_s=float(getattr(config, "mjx_control_slew_rad_s", 2.0)),
            step_clearance_m=float(getattr(config, "mjx_step_clearance_m", .020)),
            mask_fsr_obs=bool(getattr(config, "mjx_mask_fsr_obs", False)),
            mask_mag_obs=bool(getattr(config, "mjx_mask_mag_obs", False)),
            command_delay_steps=int(getattr(config, "mjx_command_delay_steps", 0)),
            imu_accel_noise=float(getattr(config, "mjx_imu_accel_noise", 0.)),
            imu_gyro_noise=float(getattr(config, "mjx_imu_gyro_noise", 0.)),
            backward_drift_weight=float(getattr(config, "mjx_backward_drift_weight", 0.0)),
            stance_knee_sag_weight=float(getattr(config, "mjx_stance_knee_sag_weight", 0.0)),
            rsi_stratified=rsi_active,
            rsi_probabilities=tuple(probabilities),
            gait_reference_csv=(None if getattr(config, "pose_rsi_json", None) else reference),
            pose_rsi_json=(reference if getattr(config, "pose_rsi_json", None) else None),
            pose_rsi_allow_falls=bool(getattr(config, "pose_rsi_allow_falls", False)),
            injected_pose_min_id=getattr(config, "mjx_injected_pose_min_id", None),
            gait_reference_signs=getattr(config, "gait_reference_signs", None),
            gait_residual=bool(getattr(config, "mjx_gait_residual", False)),
            alive_bonus=float(getattr(config, "mjx_alive_bonus", 0.)),
            standing_posture_weight=float(getattr(config, "mjx_standing_posture_weight", 0.)),
            footsteps=({k: getattr(config, "mjx_dot_" + k) for k in ("step", "radius", "hit_reward",
                        "approach_weight", "first_side", "first_from_toe", "first_behind_toe",
                        "toe_target", "max_dots", "end_on_success") if getattr(config, "mjx_dot_" + k, None) is not None}
                       if bool(getattr(config, "mjx_footsteps", False)) else None),
            reward_sizes=({k: getattr(config, "mjx_" + k) for k in ("fall_penalty", "landing_weight",
                           "first_step_bonus", "goal_bonus", "stall_out_penalty", "stall_out_seconds", "action_weight",
                           "stall_grace_seconds", "still_pays", "lateral_scale")
                           if getattr(config, "mjx_" + k, None) is not None}),
            phase_control=getattr(config, "mjx_phase_control", None),
            stabilizer=(list(config.mjx_stabilizer) if getattr(config, "mjx_stabilizer", None) else None),
            phase_obs=bool(getattr(config, "mjx_phase_obs", False)),
            scripted_phase_pool=(Path(__file__).resolve().parents[1] / str(config.mjx_scripted_phase_pool)
                                 if getattr(config, "mjx_scripted_phase_pool", None) else None),
            scripted_gait=(Path(__file__).resolve().parents[1] / str(config.mjx_scripted_gait)
                           if getattr(config, "mjx_scripted_gait", None) else None),
            imitation_csv=(Path(__file__).resolve().parents[1] / str(config.gait_reference_csv)
                           if getattr(config, "gait_reference_csv", None) else None),
            gait_reference_speed=float(getattr(config, "gait_reference_speed", 1.0)),
            rsi_reference_velocity=bool(getattr(config, "rsi_reference_velocity", True)),
            simplified_rewards=bool(getattr(config, "mjx_simplified_rewards", False)),
            reward_profile=str(getattr(config, "mjx_reward_profile", "legacy")),
            motor_torque_kgcm=getattr(config, "mjx_motor_torque_kgcm", None),
            gait_curriculum=(str(getattr(config, "mjx_task", "walking")) == "walking" and not self.full_command),
            task=str(getattr(config, "mjx_task", "walking")))
        self.training_pose_id = None
        forced_training_index = None
        configured_pose = getattr(config, "training_pose_id", None)
        if configured_pose is not None:
            # Extended 2026-10-06 alongside env.py's matching RSI relaxation,
            # and no longer training-only: forcing every reset to one exact
            # saved pose, for BOTH training and eval envs, is exactly the
            # mechanism needed to train (and then honestly measure) standing
            # balance recovery from one specific pose (the pose-balance
            # curriculum). Previously only applied when not self.evaluation;
            # no existing config used this field, so widening it breaks
            # nothing currently in use.
            if (self.raw.task not in ("walking", "standing") or not rsi_active
                    or not getattr(config, "pose_rsi_json", None)):
                raise ValueError(
                    "training_pose_id requires the walking or standing task "
                    "with active manual-pose RSI")
            from .mjx_walking.scene import training_pose_reset_index

            forced_training_index = training_pose_reset_index(
                np.asarray(self.raw.rsi_pose_ids), configured_pose)
            self.training_pose_id = int(configured_pose)
        pose_groups = getattr(config, "pose_groups", None)
        if pose_groups is not None and not self._balanced_evaluation:
            # Fixed start pose per world (e.g. 1024 = 205 standing + 205 each of poses 1-4).
            if not rsi_active or not getattr(config, "pose_rsi_json", None):
                raise ValueError("pose_groups requires active manual-pose RSI")
            from .mjx_walking.scene import grouped_pose_reset_indices

            indices = jnp.asarray(grouped_pose_reset_indices(
                np.asarray(self.raw.rsi_pose_ids), self.env_num, list(pose_groups)))
            self._reset_batch = jax.jit(lambda keys: jax.vmap(self.raw.reset)(keys, indices))
        elif self._balanced_evaluation:
            if not rsi_active or not getattr(config, "pose_rsi_json", None):
                raise ValueError("Balanced pose evaluation requires active manual-pose RSI")
            from .mjx_walking.scene import balanced_pose_reset_indices

            indices = balanced_pose_reset_indices(
                np.asarray(self.raw.rsi_pose_ids), self.env_num)
            self._evaluation_pose_indices = jnp.asarray(indices)
            self._reset_batch = jax.jit(lambda keys: jax.vmap(self.raw.reset)(
                keys, self._evaluation_pose_indices))
        elif forced_training_index is not None:
            indices = jnp.full((self.env_num,), forced_training_index, dtype=jnp.int32)
            self._reset_batch = jax.jit(lambda keys: jax.vmap(self.raw.reset)(keys, indices))
        else:
            self._reset_batch = jax.jit(jax.vmap(self.raw.reset))
        # Live pose-injection side channel (see docs/INTERACTIVE_RSI_USAGE_*.md):
        # a dedicated reset wrapper that always takes an explicit qpos/qvel/ctrl
        # batch instead of RSI-sampling one. Kept entirely separate from
        # self._reset_batch above so the normal training/eval reset path's
        # compiled behavior is untouched.
        def _forced_reset(key, qpos, qvel, ctrl):
            return self.raw.reset(key, use_forced_pose=True,
                                  forced_qpos=qpos, forced_qvel=qvel, forced_ctrl=ctrl)
        self._inject_reset_batch = jax.jit(jax.vmap(_forced_reset))
        # MJX-Warp's custom-vmap physics step works under vmap/fori_loop, but
        # wrapping it in a per-world lax.cond fails JAX custom batching.
        # Compute a batched reset and step separately, then select array leaves
        # on the GPU. This adds one forward pass per call but no host transfer.
        self._step_batch = jax.jit(jax.vmap(self.raw.step))
        def select_reset(reset_state, stepped_state, reset_mask):
            def select_leaf(reset_leaf, stepped_leaf):
                if (not hasattr(stepped_leaf, "ndim") or stepped_leaf.ndim == 0
                        or stepped_leaf.shape[0] != reset_mask.shape[0]):
                    return stepped_leaf
                mask = reset_mask.reshape((reset_mask.shape[0],) +
                                          (1,) * (stepped_leaf.ndim - 1))
                return jax.numpy.where(mask, reset_leaf, stepped_leaf)
            return jax.tree_util.tree_map(select_leaf, reset_state, stepped_state)
        self._select_reset_batch = jax.jit(select_reset)
        self.state = None
        # Set by inject_pose(); tells the next step() call to report that
        # injected state as a fresh "first" transition for all worlds instead
        # of letting the normal done-triggered reset-select logic immediately
        # overwrite it with a new random RSI sample (see inject_pose docstring).
        self._pending_injection_first = False
        self._diagnostic_renderer = None
        self._diagnostic_render_size = None
        self._diagnostic_data = None
        observation_spaces = {"proprio": spaces.Box(-np.inf, np.inf, (45,), np.float32)}
        if not self.full_command:
            observation_spaces["teacher_state"] = spaces.Box(-1., 1., (TEACHER_STATE_DIM,), np.float32)
        self._obs_space = spaces.Dict(observation_spaces)
        self._packed_obs_dim = 45 if self.full_command else 45 + TEACHER_STATE_DIM
        self._action_space = spaces.Box(-1., 1., (19,), np.float32)

    def evaluation_protocol(self):
        return {
            "version": 2, "backend": "mjx-warp", "worlds": self.env_num,
            "seed": self._evaluation_seed,
            "repeatable": self._repeatable_evaluation or self._balanced_evaluation,
            "balanced_pose_ids": self._balanced_evaluation,
            "horizon_control_steps": self.horizon,
            "control_dt_seconds": self.raw.control_dt,
            "valid_step_clearance_m": self.raw.step_clearance_m,
            "reset_source_kind": getattr(self.raw, "rsi_source_kind", "legacy"),
            "controller_sample_offset": getattr(self.raw, "gait_reference_sample_offset", None),
            "full_command": bool(getattr(self.raw, "full_command", False)),
            "controller_identity": getattr(self.raw, "controller_identity", "legacy"),
            "observation_semantics": getattr(self.raw, "observation_semantics", "legacy"),
            "observation_keys": ["proprio"] if self.full_command else ["proprio", "teacher_state"],
            "reset_joint_noise_rad": getattr(self.raw, "reset_joint_noise_rad", .005),
            "reset_velocity_noise": getattr(self.raw, "reset_velocity_noise", 0.),
            "target_forward_speed_m_s": getattr(self.raw, "target_forward_speed", .08),
            "reference_kind": getattr(self.raw, "gait_reference_kind", "none"),
            "stabilizer_enabled": getattr(self.raw, "stabilizer", None) is not None,
            "phase_observations": bool(getattr(self.raw, "phase_obs", False)),
            "gait_curriculum": bool(getattr(self.raw, "gait_curriculum", False)),
            "rsi_stratified": bool(self.raw.rsi_stratified),
            "rsi_probabilities": (np.asarray(self.raw.rsi_probabilities).tolist()
                                  if self.raw.rsi_stratified else None),
            "step_definition": "Qualified lift, supported landing and forward foot progress; dot hits are separate",
        }

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    def set_training_updates(self, updates):
        pass

    def begin_evaluation(self):
        """Repeat the balanced start schedule and RNG at every checkpoint.

        Called once by the trainer before its first reset. Legacy random-mix
        evaluations retain their original RNG behavior unless opted in.
        """
        if not self.evaluation:
            raise RuntimeError("begin_evaluation is only valid on an evaluation environment")
        if self._balanced_evaluation or getattr(self, "_repeatable_evaluation", False):
            self.rng = self.jax.random.PRNGKey(self._evaluation_seed)
            self.state = None

    def reset_pose_ids(self):
        """Return the current reset's saved-pose IDs; 0 denotes standing.

        This is evaluation metadata only. It is never part of policy input.
        """
        if self.state is None:
            raise RuntimeError("MJX teacher has not been reset")
        return torch.from_dlpack(self.state.info["rsi_pose_id"]).cpu().tolist()

    def write_live_view(self, path, reward):
        """Snapshot every world's qpos + last reward for scripts/watch_training.py."""
        if self.state is None:
            return
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as handle:  # rename so the viewer never reads a partial file
            np.savez(handle, qpos=np.asarray(self.state.pipeline_state.qpos),
                     reward=reward.detach().cpu().numpy(),
                     step=torch.from_dlpack(self.state.info["step_count"]).cpu().numpy())
        tmp.replace(path)

    def dump_world_state(self, index, path):
        """Write one world's current qpos/qvel/ctrl to an .npz file.

        Diagnostic/reference only (e.g. so a human can see the live pose before
        authoring a replacement) -- never read back into training itself.
        """
        if self.state is None:
            raise RuntimeError("MJX teacher has not been reset")
        if not 0 <= index < self.env_num:
            raise ValueError("World index out of range")
        device_data = self.state.pipeline_state
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as handle:  # rename so a reader never sees a partial file
            np.savez(handle,
                     qpos=np.asarray(device_data.qpos[index]),
                     qvel=np.asarray(device_data.qvel[index]),
                     ctrl=np.asarray(device_data.ctrl[index]),
                     step=np.asarray(int(torch.from_dlpack(self.state.info["step_count"])[index])))
        tmp.replace(path)

    def inject_pose(self, qpos, qvel=None, ctrl=None, noise_std=0.03, seed=None):
        """Force every world to reset into a human-authored pose.

        World 0 gets the exact authored pose; the remaining worlds get the
        same pose with small Gaussian noise added to the actuated joint
        angles (and the matching ctrl target), so one manual pose yields a
        small cluster of nearby starting states instead of an exact
        duplicate in every world. This fully replaces ``self.state`` --
        every world's episode restarts from here, matching a normal reset's
        bookkeeping (step_count/valid_steps/etc. all zeroed), so downstream
        per-episode logging in step() stays consistent.
        """
        if self.full_command:
            raise ValueError("Full-command student uses standing resets; external experts send ordinary actions")
        m = self.raw.m
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (m.nq,):
            raise ValueError(f"inject_pose qpos must have shape ({m.nq},), got {qpos.shape}")
        qvel = (np.zeros((m.nv,), dtype=np.float64) if qvel is None
                else np.asarray(qvel, dtype=np.float64))
        if qvel.shape != (m.nv,):
            raise ValueError(f"inject_pose qvel must have shape ({m.nv},), got {qvel.shape}")
        act_qpos = np.asarray(self.raw.act_qpos)
        nu = act_qpos.shape[0]
        ctrl = (qpos[act_qpos].copy() if ctrl is None else np.asarray(ctrl, dtype=np.float64))
        if ctrl.shape != (nu,):
            raise ValueError(f"inject_pose ctrl must have shape ({nu},), got {ctrl.shape}")
        low = np.asarray(self.raw.low, dtype=np.float64)
        high = np.asarray(self.raw.high, dtype=np.float64)

        qpos_batch = np.tile(qpos, (self.env_num, 1))
        qvel_batch = np.tile(qvel, (self.env_num, 1))
        ctrl_batch = np.tile(ctrl, (self.env_num, 1))
        if self.env_num > 1:
            rng = np.random.default_rng(seed)
            noise = rng.normal(0., noise_std, size=(self.env_num - 1, nu))
            # Same noise vector shifts both qpos and ctrl together, preserving
            # the authored pose's qpos-vs-ctrl relationship in each variant.
            qpos_batch[1:, act_qpos] = np.clip(qpos[act_qpos] + noise, low, high)
            ctrl_batch[1:] = np.clip(ctrl + noise, low, high)

        keys = self.jax.random.split(self.rng, self.env_num + 1)
        self.rng = keys[0]
        self.state = self._inject_reset_batch(
            keys[1:],
            self.jnp.asarray(qpos_batch, dtype=self.jnp.float32),
            self.jnp.asarray(qvel_batch, dtype=self.jnp.float32),
            self.jnp.asarray(ctrl_batch, dtype=self.jnp.float32))
        self._pending_injection_first = True

    def step(self, action, done):
        if action.device != self.device or done.device != self.device:
            raise ValueError("MJX teacher actions and reset mask must stay on cuda:0")
        if action.shape != (self.env_num, 19) or done.shape != (self.env_num,):
            raise ValueError("Invalid action/reset shape")
        if self.full_command and not bool(torch.all(torch.isfinite(action) & (action.abs() <= 1.))):
            raise RuntimeError("Full-command motor action must be finite and in [-1,1]")
        keys = self.jax.random.split(self.rng, self.env_num + 1)
        self.rng = keys[0]
        terminal = torch.zeros_like(done, dtype=torch.bool)
        time_limit = torch.zeros_like(done, dtype=torch.bool)
        old_state = self.state
        if old_state is None or self._pending_injection_first:
            if old_state is None:
                if not bool(done.all()):
                    raise ValueError("First MJX call must reset all worlds")
                self.state = self._reset_batch(keys[1:])
            # else: self.state already holds the just-injected pose (set by
            # inject_pose()) -- report it as this call's fresh reset output
            # instead of resampling, consuming the pending-injection flag.
            self._pending_injection_first = False
            first = torch.ones_like(done, dtype=torch.bool)
            last = torch.zeros_like(done, dtype=torch.bool)
            reward = torch.zeros(self.env_num, device=self.device)
        else:
            # Both frameworks consume/produce CUDA DLPack buffers; no tensor
            # is copied through NumPy or host RAM in the collection path.
            jax_action = self.jax.dlpack.from_dlpack(action.detach().contiguous())
            jax_done = self.jax.dlpack.from_dlpack(done.contiguous())
            reset_states = self._reset_batch(keys[1:])
            stepped_states = self._step_batch(self.state, jax_action)
            self.state = self._select_reset_batch(reset_states, stepped_states, jax_done)
            first = done.bool()
            terminal = torch.from_dlpack(self.state.done).bool() & ~first
            elapsed = torch.from_dlpack(self.state.info["step_count"])
            time_limit = (elapsed >= self.horizon) & ~terminal & ~first
            last = terminal | time_limit
            reward = torch.from_dlpack(self.state.reward).float()
        causes = self._terminal_causes(self.state.metrics, terminal, first)
        packed = torch.from_dlpack(self.state.obs).float()
        if packed.shape != (self.env_num, self._packed_obs_dim):
            raise RuntimeError("MJX teacher observation contract changed")
        torch._assert_async(torch.isfinite(packed).all(), "Nonfinite MJX teacher observation")
        torch._assert_async(torch.isfinite(reward).all(), "Nonfinite MJX teacher reward")
        result = dict(proprio=packed[:, :45],
                      reward=reward[:, None], is_first=first[:, None],
                      is_last=last[:, None], is_terminal=terminal[:, None])
        if not self.full_command:
            result["teacher_state"] = packed[:, 45:]
        def gpu(value):
            return torch.from_dlpack(value).float()
        current_steps = gpu(self.state.info["valid_steps"])
        current_scrub = gpu(self.state.info["scrub"])
        current_forward = gpu(self.state.metrics["forward_distance"])
        if old_state is None:
            step_delta = torch.zeros_like(current_steps)
            scrub_delta = torch.zeros_like(current_scrub)
            forward_delta = torch.zeros_like(current_forward)
        else:
            old_steps = gpu(old_state.info["valid_steps"])
            old_scrub = gpu(old_state.info["scrub"])
            old_forward = gpu(old_state.metrics["forward_distance"])
            step_delta = torch.where(first[:, None], 0., (current_steps - old_steps).clamp_min(0.))
            scrub_delta = torch.where(first, 0., (current_scrub - old_scrub).clamp_min(0.))
            forward_delta = torch.where(first, 0., current_forward - old_forward)
        episode_logs = {
            "diagnostic_samples": torch.ones_like(current_scrub),
            "valid_left_step": step_delta[:, 0],
            "valid_right_step": step_delta[:, 1],
            "total_valid_steps": step_delta.sum(-1),
            "alternating_steps": torch.minimum(current_steps[:, 0], current_steps[:, 1])
                - (torch.minimum(old_steps[:, 0], old_steps[:, 1]) if old_state is not None
                   else torch.zeros_like(current_scrub)),
            "fall_count": causes["physical_fall"].float(),
            "physical_fall": causes["physical_fall"].float(),
            "wall_hit": causes["wall_hit"].float(),
            "stall_out": causes["stall_out"].float(),
            "goal_reached": causes["goal_reached"].float(),
            "safety_failure_count": (causes["physical_fall"] | causes["wall_hit"]).float(),
            "terminated": terminal.float(),
            "time_limit": time_limit.float(),
            "alive_seconds": (~first).float() * self.raw.control_dt,
            "forward_distance": forward_delta,
            "scrub_distance": scrub_delta,
            "standing_stable_seconds": self._standing_seconds_delta(
                gpu(self.state.info["standing_stable_seconds"]),
                gpu(old_state.info["standing_stable_seconds"]) if old_state is not None else None,
                first),
            "backward_drift_cost": gpu(self.state.metrics["backward_drift_cost"]),
            "stance_knee_sag_cost": gpu(self.state.metrics["stance_knee_sag_cost"]),
        }
        if self.raw.reward_profile in ("step_progress_v1", "goal_corridor_v1", "goal_corridor_v2", "goal_corridor_v3", "autonomous_walk_v1"):
            old_alternations = (gpu(old_state.info["alternating_steps"]) if old_state is not None
                                else torch.zeros_like(current_scrub))
            episode_logs["alternating_steps"] = self._standing_seconds_delta(
                gpu(self.state.info["alternating_steps"]), old_alternations, first)
        current_dots = gpu(self.state.metrics["dots_reached"])
        previous_dots = (gpu(old_state.metrics["dots_reached"])
                         if old_state is not None else None)
        episode_logs["dots_reached"] = self._standing_seconds_delta(current_dots, previous_dots, first)
        episode_logs["dot_event_delta"] = (gpu(self.state.metrics["dot_event_delta"])
                                           if "dot_event_delta" in self.state.metrics
                                           else episode_logs["dots_reached"])
        episode_logs["dots_final_count_peak"] = current_dots
        for name in (
                "single_support_seconds", "left_support_seconds", "right_support_seconds",
                "left_knee_rad_mean", "right_knee_rad_mean", "root_height_m_mean",
                "body_tilt_rad_mean", "left_load_bw_mean", "right_load_bw_mean",
                "left_clearance_m_peak", "right_clearance_m_peak", "tracking_error_rad_mean",
                "target_clipped_fraction_mean", "reward_upright", "reward_load",
                "body_tilt_rad_peak", "root_drop_m_peak", "stance_knee_bend_rad_peak",
                "actuator_force_fraction_peak", "torque_saturation_fraction_mean",
                "control_slew_limited_fraction_mean",
                "reward_shift", "reward_swing", "reward_forward", "reward_touchdown", "reward_height",
                "raw_scrub_cost", "action_rate_cost", "action_magnitude_cost",
                "penalty_scrub", "penalty_action_rate", "penalty_action_magnitude",
                "penalty_backward_drift", "penalty_knee_sag", "penalty_fall",
                "reward_balance", "reward_lift_progress", "reward_alternation",
                "reward_com_center", "penalty_height_violation", "penalty_corridor",
                "standing_reward_upright", "standing_reward_height", "standing_reward_contact",
                "standing_reward_balance", "standing_reward_com_center", "standing_reward_velocity",
                "standing_penalty_slip", "standing_penalty_action", "standing_penalty_track",
                "standing_penalty_fall", "standing_streak_bonus", "standing_reward_posture",
                "reward_dot", "reward_dot_approach",
                "reward_alive", "reward_track", "reward_landing", "reward_progress", "reward_first_step", "reward_step", "reward_goal",
                "penalty_stall", "penalty_stall_out", "penalty_lateral", "penalty_heading",
                "penalty_wall"):
            episode_logs[name] = gpu(self.state.metrics[name])
        for name, value in episode_logs.items():
            result["log_" + name] = torch.where(first, 0., value).reshape(self.env_num, 1)
        return TensorDict(result, batch_size=(self.env_num,), device=self.device), last

    @staticmethod
    def _terminal_causes(metrics, terminal, first):
        names = ("physical_fall", "wall_hit", "stall_out", "goal_reached")
        if not all(name in metrics for name in names):
            return {name: (terminal & ~first if name == "physical_fall"
                           else torch.zeros_like(terminal)) for name in names}
        return {name: torch.from_dlpack(metrics[name]).bool() & terminal & ~first
                for name in names}

    @staticmethod
    def _standing_seconds_delta(current, previous, first):
        """Convert a cumulative per-episode clock into summable transition logs."""
        previous = torch.zeros_like(current) if previous is None else previous
        return torch.where(first, 0., (current - previous).clamp_min(0.))

    def render_selected(self, indices, width=192, height=192):
        """Render requested worlds only; return world-aligned diagnostic records.

        Read back selected generalized coordinates and reconstruct visual poses
        with CPU MuJoCo. These frames never enter policy observations.
        """
        import numbers

        indices = list(indices)
        if any(isinstance(index, (bool, np.bool_)) or not isinstance(index, numbers.Integral)
               or not 0 <= index < self.env_num for index in indices):
            raise ValueError("Video world indices must be integers within the environment batch")
        if len(set(indices)) != len(indices):
            raise ValueError("Video world indices must be unique")
        result = [None] * self.env_num
        if not indices:
            return result
        try:
            import mujoco

            if self.state is None:
                raise RuntimeError("MJX teacher has not been reset")
            size = (int(width), int(height))
            if min(size) < 64:
                raise ValueError("diagnostic video dimensions must be at least 64")
            if size != self._diagnostic_render_size:
                if self._diagnostic_renderer is not None:
                    self._diagnostic_renderer.close()
                self._diagnostic_renderer = mujoco.Renderer(
                    self.raw.m, width=size[0], height=size[1])
                self._diagnostic_render_size = size
            if getattr(self, "_diagnostic_data", None) is None:
                self._diagnostic_data = mujoco.MjData(self.raw.m)
            host_data = self._diagnostic_data
            device_data = self.state.pipeline_state
            for index in indices:
                try:
                    # Index before conversion: never transfer unselected worlds.
                    host_data.qpos[:] = np.asarray(device_data.qpos[index])
                    host_data.qvel[:] = np.asarray(device_data.qvel[index])
                    host_data.ctrl[:] = np.asarray(device_data.ctrl[index])
                    mujoco.mj_forward(self.raw.m, host_data)
                    self._diagnostic_renderer.update_scene(host_data, camera="external_view")
                    result[index] = {"frame": self._diagnostic_renderer.render().copy(), "error": None}
                except Exception as exc:
                    result[index] = {"frame": None, "error": f"{type(exc).__name__}: {exc}"}
        except Exception as exc:
            for index in indices:
                result[index] = {"frame": None, "error": f"{type(exc).__name__}: {exc}"}
        return result

    def render_first(self, width=192, height=192):
        """Backward-compatible single-world diagnostic frame."""
        return self.render_selected([0], width=width, height=height)[0]

    def release_video_renderer(self):
        """Release optional EGL/Torch cached memory before GPU collection."""
        if self._diagnostic_renderer is not None:
            self._diagnostic_renderer.close()
            self._diagnostic_renderer = None
            self._diagnostic_render_size = None
        self._diagnostic_data = None
        torch.cuda.empty_cache()

    def close(self):
        self.release_video_renderer()
        self.state = None
