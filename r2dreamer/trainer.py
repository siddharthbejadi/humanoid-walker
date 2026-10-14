import contextlib
import json
import pickle
import shutil
import time

import numpy as np
import torch

import tools
from training_budget import validate_learning_budget
from checkpoint_selection import checkpoint_rank, walking_trial_success, finish_trial_success
from gait_guard import preservation_result


class OnlineTrainer:
    # Fixed order of the eight corner FSR channels appended to proprio, matching
    # FSR_INTERFACE.md (LHO, LHI, LTO, LTI, RHO, RHI, RTO, RTI).
    FSR_CHANNEL_NAMES = ("left_heel_outer", "left_heel_inner", "left_toe_outer", "left_toe_inner",
                         "right_heel_outer", "right_heel_inner", "right_toe_outer", "right_toe_inner")

    def __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.logdir = logdir
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.video_pred_log = bool(config.video_pred_log)
        self.params_hist_log = bool(config.params_hist_log)
        self.batch_length = int(config.batch_length)
        batch_steps = int(config.batch_size * config.batch_length)
        # train_ratio is based on data steps rather than environment steps.
        self._updates_needed = tools.Every(batch_steps / config.train_ratio * config.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._action_repeat = config.action_repeat

        # Periodic checkpointing -- previously only a single final "latest.pt" was ever
        # saved (in train.py, after the whole loop returns), which throws away the entire
        # behavioral history of a run: no way to see what the policy looked like partway
        # through, and no way to recover the best-performing point if it later regresses.
        # checkpoint_interval/best_metric_key are optional config fields (getattr falls
        # back to these defaults for existing configs that don't define them, same pattern
        # already used for enable_vision/scene_path in envs/__init__.py's make_env()).
        self.checkpoint_interval = int(getattr(config, "checkpoint_interval", 25000))
        self._should_checkpoint = tools.Every(self.checkpoint_interval)
        # Optional bounded retention for long runs on small instance disks.
        # Only periodic checkpoints in this run's logdir are rotated; best.pt,
        # best_safe.pt, and checkpoints from other runs are never touched.
        self.checkpoint_keep_last = int(getattr(config, "checkpoint_keep_last", 0))
        if self.checkpoint_keep_last < 0:
            raise ValueError("checkpoint_keep_last must be nonnegative")
        self.train_video = bool(getattr(config, "train_video", True))
        self.best_metric_key = str(getattr(config, "best_metric_key", "score"))
        self._best_metric_value = -float("inf")
        self.best_safety_metric_key = getattr(config, "best_safety_metric_key", None)
        self.best_tiebreak_metric_key = str(getattr(config, "best_tiebreak_metric_key", "score"))
        self._best_rank = None
        self._best_safe_rank = None
        self._last_selection = None
        self._protected_best = set()
        self.checkpoint_order = str(getattr(config, "checkpoint_order", "safety_first"))
        checkpoint_rank(0., 0., 0., self.checkpoint_order)
        self.save_best_safe = bool(getattr(config, "save_best_safe", False))
        if self.save_best_safe and self.best_safety_metric_key not in ("fall_count", "safety_failure_count"):
            raise ValueError("save_best_safe requires fall_count or safety_failure_count as the safety metric")
        self.last_train_step = 0
        self.encoder_training_aid = bool(getattr(config, "encoder_training_aid", False))
        self.encoder_boundaries = (
            int(getattr(config, "encoder_full_updates", 500)),
            int(getattr(config, "encoder_off_updates", 1000)),
        )
        if self.encoder_training_aid and not 0 <= self.encoder_boundaries[0] < self.encoder_boundaries[1]:
            raise ValueError("Invalid encoder schedule")
        self.update_count = 0
        self.training_contract = None
        self.imitation_dataset = None
        self.imitation_settings = None
        self.imitation_metadata = None
        self.imitation_rng = None
        self.stop_after_optimizer_updates = int(getattr(config, "stop_after_optimizer_updates", 0))
        if self.stop_after_optimizer_updates < 0:
            raise ValueError("stop_after_optimizer_updates must be nonnegative")
        self._stop_requested = False
        self._stop_reason = None
        # Default timings measure host wall time, including enqueuing CUDA work.
        # Synchronized phase timings are opt-in because they perturb throughput.
        self.profile_synchronize = bool(getattr(config, "profile_synchronize", False))
        self._phase_seconds = {}
        # Opt-in production guard: collection-only runs must never masquerade
        # as a successful learning continuation.
        self.require_learning = bool(getattr(config, "require_learning", False))
        self.learning_start_deadline_steps = int(getattr(config, "learning_start_deadline_steps", 0))
        self.gait_guard_enabled = bool(getattr(config, "gait_guard_enabled", False))
        self.gait_guard_eval_updates = int(getattr(config, "gait_guard_eval_updates", 8))
        self.gait_guard_thresholds = {
            "minimum_initial_success": float(getattr(config, "gait_guard_minimum_initial_success", .5)),
            "success_fraction": float(getattr(config, "gait_guard_success_fraction", .8)),
            "step_fraction": float(getattr(config, "gait_guard_step_fraction", .7)),
            "extra_failure_rate": float(getattr(config, "gait_guard_extra_failure_rate", 2. / 32.)),
        }
        self._gait_baseline = None
        self._gait_last_eval_updates = None
        self._gait_protocol = None
        if self.gait_guard_enabled and (self.gait_guard_eval_updates <= 0 or self.eval_episode_num != 32):
            raise ValueError("Gait guard requires a positive update interval and 32 evaluation worlds")
        self.eval_video = bool(getattr(config, "eval_video", False))
        # MP4 export can remain enabled when an installed TensorBoard/NumPy
        # pair cannot encode videos into event files.
        self.eval_video_tensorboard = bool(getattr(config, "eval_video_tensorboard", True))
        self.eval_video_every = int(getattr(config, "eval_video_every", self.eval_every))
        self.eval_video_stride = int(getattr(config, "eval_video_stride", 2))
        self.eval_video_width = int(getattr(config, "eval_video_width", 192))
        self.eval_video_height = int(getattr(config, "eval_video_height", 192))
        self.eval_video_fps = int(getattr(config, "eval_video_fps", 25))
        self.eval_video_save_mp4 = bool(getattr(config, "eval_video_save_mp4", False))
        self.eval_video_final_only = bool(getattr(config, "eval_video_final_only", False))
        self.eval_video_one_per_pose = bool(getattr(config, "eval_video_one_per_pose", False))
        # Routine evaluation can be shorter than a very long navigation episode.
        # A separate full-distance rollout should be used for final qualification.
        self.eval_max_steps = int(getattr(config, "eval_max_steps", 0))
        if self.eval_max_steps < 0:
            raise ValueError("eval_max_steps must be nonnegative")
        if self.eval_video and (
                self.eval_video_every <= 0 or self.eval_video_stride <= 0 or
                self.eval_video_width < 64 or self.eval_video_height < 64 or
                self.eval_video_fps <= 0):
            raise ValueError("Invalid evaluation video configuration")

    def _selection_settings(self):
        protocol = (self.eval_envs.evaluation_protocol()
                    if self.eval_envs is not None and hasattr(self.eval_envs, "evaluation_protocol")
                    else None)
        return {"metric": self.best_metric_key, "safety": self.best_safety_metric_key,
                "tiebreak": self.best_tiebreak_metric_key, "order": self.checkpoint_order,
                "evaluation": protocol, "eval_max_steps": self.eval_max_steps}

    def restore_checkpoint_selection(self, checkpoint):
        state = checkpoint.get("checkpoint_selection")
        existing = {name for name in ("best.pt", "best_safe.pt")
                    if (self.logdir / name).exists()}
        if state is None:
            self._protected_best = existing
            if existing:
                print("Resume checkpoint has no recorded ranking; preserving existing "
                      "best checkpoints. Use a new run directory for a new ranking history.")
            return
        if state.get("version") != 1:
            raise ValueError("Unsupported checkpoint selection metadata")
        if state.get("settings") != self._selection_settings():
            if existing:
                raise ValueError("Checkpoint selection protocol changed; use a new run directory "
                                 "to preserve existing best checkpoints")
            print("Checkpoint selection protocol changed; starting ranking in the new run directory")
            return
        for name, width in (("best_rank", 3), ("best_safe_rank", 2)):
            value = state.get(name)
            if value is not None and (len(value) != width or not np.isfinite(value).all()):
                raise ValueError(f"Invalid checkpoint selection {name}")
            setattr(self, "_" + name, tuple(value) if value is not None else None)
        self._last_selection = state.get("last_evaluation")
        self._protected_best = set(state.get("protected_best", [])) & existing
        # A crash can leave best.pt newer than latest.pt. Merge its recorded rank
        # before judging the first resumed evaluation.
        for name in existing - self._protected_best:
            try:
                candidate = torch.load(self.logdir / name, map_location="cpu", weights_only=True)
                candidate_state = candidate.get("checkpoint_selection", {})
                if candidate_state.get("settings") != self._selection_settings():
                    self._protected_best.add(name)
                else:
                    key = "best_rank" if name == "best.pt" else "best_safe_rank"
                    value = candidate_state.get(key)
                    width = 3 if key == "best_rank" else 2
                    if value is None or len(value) != width or not np.isfinite(value).all():
                        self._protected_best.add(name)
                    else:
                        old = getattr(self, "_" + key)
                        setattr(self, "_" + key, max(old, tuple(value)) if old is not None else tuple(value))
                del candidate
            except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError,
                    IndexError, EOFError, pickle.UnpicklingError):
                self._protected_best.add(name)
                print(f"Could not verify existing {name} ranking; preserving that checkpoint")

    def _checkpoint_selection_state(self):
        return {"version": 1, "settings": self._selection_settings(),
                "best_rank": list(self._best_rank) if self._best_rank is not None else None,
                "best_safe_rank": (list(self._best_safe_rank)
                                   if self._best_safe_rank is not None else None),
                "last_evaluation": self._last_selection,
                "protected_best": sorted(self._protected_best)}

    def _evaluation_artifact_suffix(self, index):
        updates = f"_update_{self.update_count:06d}" if self.gait_guard_enabled else ""
        return updates + ("" if index == 0 else f"_env_{index:02d}")

    def eval(self, agent, train_step):
        """Run evaluation episodes.

        For CPU-based environments (``ParallelEnv``), stepping is executed on
        CPU and observations are moved to GPU asynchronously.  For GPU-resident
        environments (``IsaacLabVecEnv``), no device transfer is needed —
        ``.to()`` is a no-op when source and target devices match.
        """
        print("Evaluating the policy...")
        envs = self.eval_envs
        if hasattr(envs, "begin_evaluation"):
            envs.begin_evaluation()
        agent.eval()
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=agent.device)
        steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        log_metrics = {}
        start_pose_ids = None
        # cache is only used for video logging / open-loop prediction.
        cache = []
        capture_video = (
            # eval() is itself called on the configured evaluation schedule.
            # Parallel stepping advances in batches, so train_step will usually
            # pass (rather than equal) an exact multiple of eval_video_every.
            self.eval_video and
            (not self.eval_video_final_only or train_step >= self.steps) and
            (hasattr(envs, "render_selected") or hasattr(envs, "render_all") or hasattr(envs, "render_first"))
        )
        # Keep one independent stream per fixed-seed evaluation environment.
        # In particular, a falling seed must not be hidden by env 0's video.
        video_frames = [[] for _ in range(envs.env_num)]
        camera_frames = [[] for _ in range(envs.env_num)]
        fsr_records = [[] for _ in range(envs.env_num)]
        render_enabled = [capture_video for _ in range(envs.env_num)]
        video_frame_index = 0
        video_indices = None
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while not once_done.all():
            steps += ~done * ~once_done
            # Step environments.  Each env backend handles device placement
            # internally (ParallelEnv converts to CPU, IsaacLabVecEnv keeps
            # on GPU).  The .to() calls below are no-ops when the data is
            # already on agent.device.
            # (B, A), (B,)
            trans, step_done = envs.step(act.detach(), done)
            if start_pose_ids is None and hasattr(envs, "reset_pose_ids"):
                start_pose_ids = envs.reset_pose_ids()
            if capture_video and video_indices is None:
                if self.eval_video_one_per_pose:
                    if start_pose_ids is None or len(start_pose_ids) != envs.env_num:
                        raise ValueError("One-video-per-pose requires recorded reset pose IDs")
                    # Preserve the first complete episode of each starting pose.
                    video_indices = set()
                    seen_poses = set()
                    for index, pose_id in enumerate(start_pose_ids):
                        if pose_id not in seen_poses:
                            video_indices.add(index)
                            seen_poses.add(pose_id)
                else:
                    video_indices = set(range(envs.env_num))
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            # (B,)
            done = step_done.to(agent.device)

            # Record every fixed-seed environment's first episode, including
            # its terminal frame. Faster-falling envs reset while the remaining
            # evaluations finish; ``once_done`` prevents those reset frames from
            # masquerading as recovery in the failed seed's video.
            capture_indices = [
                index for index in range(envs.env_num)
                if (index in video_indices and render_enabled[index] and
                    not bool(once_done[index]) and
                    (video_frame_index % self.eval_video_stride == 0 or
                     bool(done[index])))
            ] if capture_video else []
            if capture_indices:
                if "proprio" in trans:
                    proprio_np = tools.to_np(trans["proprio"])
                    n_fsr = len(self.FSR_CHANNEL_NAMES)
                    has_sole_height = hasattr(envs, "envs")
                    for index in capture_indices:
                        vec = proprio_np[index]
                        vec = vec[0] if vec.ndim > 1 else vec
                        if vec.shape[-1] >= n_fsr:
                            extra = []
                            if has_sole_height:
                                # Ground-truth clearance (m) vs. floor; negative means
                                # penetration. Cheap to compute, invaluable for telling
                                # a real physics bug apart from a fall pose that just
                                # renders oddly at low resolution.
                                try:
                                    left = envs.envs[index].foot_ground_clearance("left_foot")()
                                    right = envs.envs[index].foot_ground_clearance("right_foot")()
                                    extra = [float(left), float(right)]
                                except Exception:
                                    extra = []
                            fsr_records[index].append(
                                (int(steps[index]), vec[-n_fsr:].astype(float).tolist(), extra))
                if "image" in trans:
                    for index in capture_indices:
                        camera_frames[index].append(
                            tools.to_np(trans["image"][index]).copy())
                if hasattr(envs, "render_selected"):
                    rendered_envs = envs.render_selected(
                        capture_indices, width=self.eval_video_width,
                        height=self.eval_video_height)
                elif hasattr(envs, "render_all"):
                    rendered_envs = envs.render_all(
                        width=self.eval_video_width,
                        height=self.eval_video_height,
                    )
                else:
                    # Backward-compatible fallback for a custom backend that
                    # only implements the historical env-0 renderer.
                    rendered_envs = [envs.render_first(
                        width=self.eval_video_width,
                        height=self.eval_video_height,
                    )] + [None] * (envs.env_num - 1)
                for index in capture_indices:
                    rendered = rendered_envs[index]
                    if rendered is None or not render_enabled[index]:
                        continue
                    if rendered["error"]:
                        if index == 0:
                            print(f"Evaluation video disabled: {rendered['error']}")
                        else:
                            print(f"Evaluation video env {index} disabled: "
                                  f"{rendered['error']}")
                        render_enabled[index] = False
                    else:
                        video_frames[index].append(rendered["frame"])
            video_frame_index += 1

            # Store transition.
            # We keep the observation and the action that produced it together.
            trans["action"] = act
            if self.video_pred_log and len(cache) < self.batch_length:
                cache.append(trans.clone())
            # (B, A)
            act, agent_state = agent.act(trans, agent_state, eval=True)
            returns += trans["reward"][:, 0] * ~once_done
            for key, value in trans.items():
                if key.startswith("log_"):
                    if key not in log_metrics:
                        log_metrics[key] = torch.zeros_like(returns)
                    if key.endswith("_peak"):
                        log_metrics[key] = torch.maximum(
                            log_metrics[key], value[:, 0] * ~once_done)
                    else:
                        log_metrics[key] += value[:, 0] * ~once_done
            once_done |= done
            if self.eval_max_steps:
                once_done |= steps >= self.eval_max_steps
        # dict of (B, T, *)
        cache = torch.stack(cache, dim=1) if len(cache) else None
        self.logger.scalar("episode/eval_score", returns.mean())
        self.logger.scalar("episode/eval_length", steps.to(torch.float32).mean())
        eval_summary = {"score": float(returns.mean()), "length": float(steps.to(torch.float32).mean())}
        for key, value in log_metrics.items():
            if key.endswith("_mean") and "log_diagnostic_samples" in log_metrics:
                value = value / log_metrics["log_diagnostic_samples"].clamp_min(1.)
                log_metrics[key] = value
            if key == "log_success":
                value = torch.clip(value, max=1.0)  # make sure 1.0 for success episode
            self.logger.scalar(f"episode/eval_{key[4:]}", value.mean())
            eval_summary[key[4:]] = float(value.mean())
        walking_keys = ("fall_count", "alive_seconds", "total_valid_steps",
                        "alternating_steps", "valid_left_step", "valid_right_step", "forward_distance")
        if "log_standing_stable_seconds" in log_metrics:
            walking_keys += ("standing_stable_seconds",)
        walking_keys += tuple(key[4:] for key in log_metrics
                              if key.endswith(("_mean", "_peak"))
                              or key.startswith(("log_reward_", "log_penalty_"))
                              or key in ("log_single_support_seconds", "log_left_support_seconds",
                                         "log_right_support_seconds", "log_scrub_distance",
                                         "log_raw_scrub_cost", "log_action_rate_cost",
                                         "log_action_magnitude_cost"))
        walking_keys += tuple(key for key in (
            "physical_fall", "wall_hit", "stall_out", "goal_reached",
            "safety_failure_count", "terminated", "time_limit", "dots_reached", "dot_event_delta")
            if "log_" + key in log_metrics)
        if all("log_" + key in log_metrics for key in walking_keys):
            walking_keys = tuple(dict.fromkeys(walking_keys))
            trial_values = {key: log_metrics["log_" + key].detach().cpu().tolist()
                            for key in walking_keys}
            trials = [{key: trial_values[key][i] for key in walking_keys}
                      for i in range(envs.env_num)]
            if start_pose_ids is not None:
                for index, trial in enumerate(trials):
                    pose_id = int(start_pose_ids[index])
                    trial["start_pose_id"] = pose_id
                    trial["start_pose"] = ("standing" if pose_id == 0
                                           else f"manual_pose_{pose_id}")
            eval_seeds = getattr(envs, "seeds", None)
            if eval_seeds is not None:
                for index, trial in enumerate(trials):
                    trial["seed"] = int(eval_seeds[index])
            for index, trial in enumerate(trials):
                trial["evaluation_world"] = index
                reasons = [name for name in ("physical_fall", "wall_hit", "stall_out", "goal_reached")
                           if trial.get(name, 0) > 0]
                if not reasons and trial.get("time_limit", 0) > 0:
                    reasons = ["time_limit"]
                if not reasons:
                    reasons = ["evaluation_limit" if self.eval_max_steps and
                               int(steps[index]) >= self.eval_max_steps and not bool(done[index])
                               else "legacy_terminal"]
                trial["termination_reasons"] = reasons
                trial["control_steps"] = int(steps[index])
                trial["score"] = float(returns[index])
                trial["evaluation_truncated"] = reasons in (["evaluation_limit"], ["time_limit"])
            rate = sum(walking_trial_success(trial) for trial in trials) / len(trials)
            eval_summary["walking_success_rate"] = rate
            self.logger.scalar("episode/eval_walking_success_rate", rate)
            finish_rate = sum(finish_trial_success(trial) for trial in trials) / len(trials)
            eval_summary["goal_success_rate"] = finish_rate
            self.logger.scalar("episode/eval_goal_success_rate", finish_rate)
            with (self.logdir / "walking_evaluations.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": int(train_step), "walking_success_rate": rate,
                                          "schema_version": 2,
                                          **({"optimizer_updates": int(self.update_count)} if self.gait_guard_enabled else {}),
                                         "goal_success_rate": finish_rate,
                                          "training_contract": self.training_contract,
                                          "protocol": {"eval_max_control_steps": self.eval_max_steps,
                                              "environment": (envs.evaluation_protocol() if
                                                  hasattr(envs, "evaluation_protocol") else None)},
                                         "trials": trials}) + "\n")
        for index, frames in enumerate(camera_frames):
            if not frames:
                continue
            camera_video = np.stack(frames, axis=0)[None]
            camera_tag = ("eval_camera_pov" if index == 0 else
                          f"eval_camera_pov/env_{index:02d}")
            if self.eval_video_tensorboard:
                self.logger.video(camera_tag, camera_video, fps=self.eval_video_fps)
            if self.eval_video_save_mp4:
                try:
                    import imageio.v2 as imageio

                    video_dir = self.logdir / "eval_videos"
                    video_dir.mkdir(parents=True, exist_ok=True)
                    suffix = self._evaluation_artifact_suffix(index)
                    camera_path = video_dir / f"camera_step_{train_step:07d}{suffix}.mp4"
                    imageio.mimsave(
                        camera_path, camera_video[0], fps=self.eval_video_fps,
                        codec="libx264", quality=7, macro_block_size=None,
                    )
                    if index == 0:
                        print(f"Camera POV MP4: {camera_path}")
                    else:
                        print(f"Camera POV MP4 env {index}: {camera_path}")
                except Exception as exc:
                    if index == 0:
                        print(f"Camera POV MP4 disabled for this run: "
                              f"{type(exc).__name__}: {exc}")
                    else:
                        print(f"Camera POV MP4 env {index} disabled for this run: "
                              f"{type(exc).__name__}: {exc}")
        for index, records in enumerate(fsr_records):
            if not records:
                continue
            fsr_dir = self.logdir / "fsr_logs"
            fsr_dir.mkdir(parents=True, exist_ok=True)
            suffix = self._evaluation_artifact_suffix(index)
            fsr_path = fsr_dir / f"step_{train_step:07d}{suffix}.csv"
            has_clearance = any(extra for _, _, extra in records)
            columns = list(self.FSR_CHANNEL_NAMES)
            if has_clearance:
                columns += ["left_sole_clearance_m", "right_sole_clearance_m"]
            with fsr_path.open("w", encoding="utf-8") as handle:
                handle.write("env_step," + ",".join(columns) + "\n")
                for env_step, values, extra in records:
                    row = values + (extra if has_clearance else [])
                    handle.write(f"{env_step}," + ",".join(f"{v:.6f}" for v in row) + "\n")
            if index == 0:
                print(f"FSR diagnostic CSV: {fsr_path}")
            else:
                print(f"FSR diagnostic CSV env {index}: {fsr_path}")
        for index, frames in enumerate(video_frames):
            if not frames:
                continue
            video = np.stack(frames, axis=0)[None]
            video_tag = ("episode/eval_policy_video" if index == 0 else
                         f"episode/eval_policy_video/env_{index:02d}")
            frame_tag = ("episode/eval_video_frames" if index == 0 else
                         f"episode/eval_video_frames/env_{index:02d}")
            if self.eval_video_tensorboard:
                self.logger.video(video_tag, video, fps=self.eval_video_fps)
            self.logger.scalar(frame_tag, len(frames))
            seed_text = ""
            eval_seeds = getattr(envs, "seeds", None)
            if eval_seeds is not None:
                seed_text = f" seed {eval_seeds[index]}"
            if index == 0:
                print(f"Evaluation video: {len(frames)} frames at step {train_step} "
                      "(TensorBoard Images: episode/eval_policy_video)")
            else:
                print(f"Evaluation video env {index}{seed_text}: {len(frames)} frames "
                      f"at step {train_step} (TensorBoard Images: {video_tag})")
            if self.eval_video_save_mp4:
                try:
                    import imageio.v2 as imageio

                    video_dir = self.logdir / "eval_videos"
                    video_dir.mkdir(parents=True, exist_ok=True)
                    suffix = self._evaluation_artifact_suffix(index)
                    video_path = video_dir / f"step_{train_step:07d}{suffix}.mp4"
                    imageio.mimsave(
                        video_path, frames, fps=self.eval_video_fps,
                        codec="libx264", quality=7, macro_block_size=None,
                    )
                    video_path.with_suffix(".json").write_text(json.dumps({
                        "training_step": train_step, "evaluation_world": index,
                        **({"optimizer_updates": int(self.update_count)} if self.gait_guard_enabled else {}),
                        "start_pose_id": (int(start_pose_ids[index])
                                          if start_pose_ids is not None else None),
                        "frames": len(frames), "fps": self.eval_video_fps,
                        "scope": "First evaluation episode, including terminal frame; not a walking certificate",
                    }, indent=2) + "\n", encoding="utf-8")
                    if index == 0:
                        print(f"Evaluation MP4: {video_path}")
                    else:
                        print(f"Evaluation MP4 env {index}: {video_path}")
                except Exception as exc:
                    # A missing system codec must not kill an otherwise healthy run;
                    # the same frames remain available in TensorBoard.
                    if index == 0:
                        print(f"Evaluation MP4 disabled for this run: "
                              f"{type(exc).__name__}: {exc}")
                    else:
                        print(f"Evaluation MP4 env {index} disabled for this run: "
                              f"{type(exc).__name__}: {exc}")
        if self.video_pred_log and cache is not None:
            initial = agent.get_initial_state(1)
            self.logger.video(
                "eval_open_loop",
                tools.to_np(
                    agent.video_pred(
                        cache[:1],  # give only first batch
                        (initial["stoch"], initial["deter"]),
                    )
                ),
            )
        self._write_log(agent, train_step)
        if hasattr(envs, "release_video_renderer"):
            envs.release_video_renderer()
        agent.train()
        return eval_summary

    def _evaluate_and_checkpoint(self, agent, step):
        # Evaluation must use the same encoder/action-authority schedule point as
        # training; otherwise ramped-control checkpoints are judged under a
        # different physical action mapping.
        if self.eval_envs is not None and hasattr(self.eval_envs, "set_training_updates"):
            self.eval_envs.set_training_updates(self.update_count)
        with self._timed_phase("evaluation", agent):
            summary = self.eval(agent, step)
        if self.gait_guard_enabled:
            guard = self._record_gait_guard(summary, step)
            if guard["phase"] == "baseline":
                self.save_checkpoint(agent, "gait_baseline.pt", step=step)
            if not guard["passed"]:
                return summary
        metric = summary.get(self.best_metric_key)
        tie = summary.get(self.best_tiebreak_metric_key)
        safety = (summary.get(self.best_safety_metric_key)
                  if self.best_safety_metric_key else 0.)
        rank = checkpoint_rank(metric, safety, tie, self.checkpoint_order)
        self._last_selection = {"step": int(step), "metric": float(metric),
                                "safety": float(safety), "tiebreak": float(tie)}
        if self._best_rank is None or rank > self._best_rank:
            self._best_rank = rank
            self._best_metric_value = float(metric)
            if "best.pt" not in self._protected_best:
                self.save_checkpoint(agent, "best.pt", step=step)
        # This is only no-fall on the evaluation seeds, not hardware safety.
        if self.save_best_safe and float(safety) == 0.:
            safe_rank = (float(metric), float(tie))
            if self._best_safe_rank is None or safe_rank > self._best_safe_rank:
                self._best_safe_rank = safe_rank
                if "best_safe.pt" not in self._protected_best:
                    self.save_checkpoint(agent, "best_safe.pt", step=step)
        return summary

    def _record_gait_guard(self, summary, step):
        protocol = self._selection_settings()["evaluation"]
        if (not isinstance(protocol, dict) or protocol.get("worlds") != 32 or
                protocol.get("repeatable") is not True or
                protocol.get("horizon_control_steps") != 1000 or self.eval_max_steps != 1000):
            raise ValueError("Gait guard requires repeatable 32-world, 20-second evaluations")
        if self._gait_protocol is not None and protocol != self._gait_protocol:
            raise ValueError("Gait guard evaluation protocol changed during recovery")
        self._gait_protocol = protocol
        result = preservation_result(self._gait_baseline, summary, **self.gait_guard_thresholds)
        result.update(step=int(step), optimizer_updates=int(self.update_count),
                      protocol=protocol, thresholds=self.gait_guard_thresholds,
                      inference_runtime={
                          "torch_version": str(torch.__version__),
                          "float32_matmul_precision": torch.get_float32_matmul_precision(),
                          "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                          "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                      },
                      walking_qualified=False)
        self._gait_last_eval_updates = self.update_count
        if result["phase"] == "baseline":
            self._gait_baseline = result["baseline"]
        with (self.logdir / "gait_guard.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, allow_nan=False) + "\n")
        temporary = self.logdir / "gait_guard_report.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(self.logdir / "gait_guard_report.json")
        if not result["passed"]:
            self.request_stop("gait_baseline_rejected" if result["phase"] == "baseline" else "gait_regression")
            print(f"GAIT GUARD STOP: {result['failures']}", flush=True)
        return result

    def save_checkpoint(self, agent, name, step=0):
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            # Absolute training step this checkpoint was taken at (NOT the replay
            # buffer's own transition count) -- read back by train.py's resume_from
            # so a resumed run's step numbering (and eval/checkpoint/log cadence)
            # continues from where this checkpoint left off, rather than from 0.
            # The replay buffer's contents themselves are NOT saved/restored -- a
            # resumed run starts with an empty buffer and briefly refills before
            # updates resume, same cost as the normal pretraining warm-up.
            "step": int(step),
            "optimizer_updates": int(self.update_count),
            "training_contract": self.training_contract,
            "stop_reason": self._stop_reason,
            "checkpoint_selection": self._checkpoint_selection_state(),
            "imitation": self.imitation_metadata,
            "model_config": getattr(agent, "export_model_config", None),
        }
        destination = self.logdir / name
        temporary = self.logdir / (name + ".tmp")
        with self._timed_phase("checkpoint", agent):
            torch.save(items_to_save, temporary)
            temporary.replace(destination)

    def _check_pose_injection(self, agent, step):
        """Live pose-injection side channel (docs/INTERACTIVE_RSI_USAGE_*.md).

        Polled once per outer training-loop iteration -- a cheap path-exists
        check in the common (no request pending) case. A human drops
        ``pose_injection_request.flag`` into this run's logdir (written by
        interactive_rsi.py's inject mode) to pause training; this method then
        blocks until ``injection_edited_state.npz`` appears, snapshots a
        backup, applies it to every world via envs.inject_pose, and returns
        True so the caller resets the loop's own done/returns/lengths/
        agent-state bookkeeping to match a genuine fresh episode start.
        """
        envs = self.train_envs
        if not hasattr(envs, "inject_pose"):
            return False
        request_path = self.logdir / "pose_injection_request.flag"
        if not request_path.exists():
            return False
        print(f"\n*** Pose injection requested at step {step}; pausing training. "
              f"Waiting for injection_edited_state.npz in {self.logdir} ***", flush=True)
        if hasattr(envs, "dump_world_state"):
            try:
                envs.dump_world_state(0, self.logdir / "injection_current_state.npz")
                print("Dumped world 0's current state to injection_current_state.npz "
                      "for reference.", flush=True)
            except Exception as exc:
                print(f"Could not dump current state (non-fatal): "
                      f"{type(exc).__name__}: {exc}", flush=True)
        edited_path = self.logdir / "injection_edited_state.npz"
        waited = 0
        while not edited_path.exists():
            if self._stop_requested:
                request_path.unlink(missing_ok=True)
                print("Pose injection cancelled: stop requested while waiting.", flush=True)
                return False
            time.sleep(1.0)
            waited += 1
            if waited % 30 == 0:
                print(f"Still waiting for injection_edited_state.npz ({waited}s)...", flush=True)
        backup_dir = self.logdir / "injections" / time.strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        self.save_checkpoint(agent, "latest.pt", step=step)
        shutil.copy(self.logdir / "latest.pt", backup_dir / "pre_injection_latest.pt")
        shutil.copy(edited_path, backup_dir / "injection_edited_state.npz")
        current_ref = self.logdir / "injection_current_state.npz"
        if current_ref.exists():
            shutil.copy(current_ref, backup_dir / "injection_current_state.npz")
        data = np.load(edited_path)
        noise_std = float(data["noise_std"]) if "noise_std" in data else 0.03
        seed = int(data["seed"]) if "seed" in data else None
        envs.inject_pose(
            data["qpos"],
            qvel=data["qvel"] if "qvel" in data else None,
            ctrl=data["ctrl"] if "ctrl" in data else None,
            noise_std=noise_std, seed=seed)
        with (self.logdir / "injections.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(
                step=int(step), timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                noise_std=noise_std, backup_dir=str(backup_dir))) + "\n")
        request_path.unlink(missing_ok=True)
        edited_path.unlink(missing_ok=True)
        print(f"*** Pose injection applied at step {step}; all {envs.env_num} worlds "
              f"reset to the authored pose (+noise); resuming training. "
              f"Backup: {backup_dir} ***\n", flush=True)
        return True

    def request_stop(self, reason="requested"):
        """Signal-safe request; saving happens only after a completed update."""
        self._stop_reason = str(reason)
        self._stop_requested = True

    @contextlib.contextmanager
    def _timed_phase(self, name, agent):
        synchronize = self.profile_synchronize and torch.device(agent.device).type == "cuda"
        if synchronize:
            torch.cuda.synchronize(agent.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            if synchronize:
                torch.cuda.synchronize(agent.device)
            self._phase_seconds[name] = self._phase_seconds.get(name, 0.) + time.perf_counter() - started

    def _log_performance(self, step):
        elapsed = max(time.perf_counter() - self._run_started, 1e-9)
        transitions = (step - self._run_initial_step) / self._action_repeat
        updates = self.update_count - self._run_initial_updates
        self.logger.scalar("perf/wall_seconds", elapsed)
        self.logger.scalar("perf/environment_transitions_per_second", transitions / elapsed)
        self.logger.scalar("perf/optimizer_updates_per_second", updates / elapsed)
        self.logger.scalar("perf/phase_timing_synchronized", self.profile_synchronize)
        for name, seconds in self._phase_seconds.items():
            self.logger.scalar(f"perf/{name}_seconds", seconds)

    def _write_log(self, agent, step, **kwargs):
        with self._timed_phase("logging", agent):
            self.logger.write(step, **kwargs)

    def _rotate_periodic_checkpoints(self):
        if self.checkpoint_keep_last <= 0:
            return
        checkpoints = sorted(
            self.logdir.glob("step_*.pt"),
            key=lambda path: int(path.stem[5:]) if path.stem[5:].isdigit() else -1,
        )
        for old in checkpoints[:-self.checkpoint_keep_last]:
            if old.stem[5:].isdigit():
                old.unlink()

    def begin(self, agent, step_offset=0, update_offset=0):
        """Main online training loop.

        For CPU-based environments the loop overlaps CPU stepping and GPU
        model execution via pinned-memory async H2D transfers.  For
        GPU-resident environments (IsaacLab) no transfer is needed —
        ``.to()`` is a no-op when the data is already on the target device.

        ``step_offset`` is the absolute step count carried over from a resumed
        checkpoint (0 for a fresh run). It is added on top of every step number
        used for the ``self.steps`` budget, eval/checkpoint/log cadence, and
        TensorBoard x-axis, so those all read as continuous absolute training
        steps. It is deliberately NOT added into the replay-buffer-relative
        ``step`` variable below, which gates when enough fresh data has
        accumulated to start updating -- that must stay relative to this
        process's own (empty-at-start) buffer regardless of prior history.
        """
        envs = self.train_envs
        video_cache = []
        step = self.replay_buffer.count() * self._action_repeat
        self.update_count = int(update_offset)
        initial_updates = self.update_count
        self.last_train_step = int(step_offset + step)
        self._run_started = time.perf_counter()
        self._run_initial_step = self.last_train_step
        self._run_initial_updates = initial_updates
        first_update_verified = False
        if self.require_learning:
            warmup = validate_learning_budget(
                envs.env_num, self.batch_length, self._action_repeat,
                self.replay_buffer.max_size, self.steps - step_offset)
            print(f"Learning preflight: {envs.env_num} worlds; fresh-replay warm-up "
                  f"<= {warmup} counted steps; starting optimizer_updates={initial_updates}", flush=True)
        envs.set_training_updates(self.update_count)
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        episode_ids = torch.arange(
            envs.env_num, dtype=torch.int32, device=agent.device
        )  # Kept constant so short episodes (< batch_length) remain sampable; RSSM resets via is_first.
        train_metrics = {}
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        injection_report_steps = 0
        last_live_view = 0.
        while step_offset + step < self.steps and not self._stop_requested:
            # Evaluation
            evaluation_due = (self._gait_last_eval_updates is None if self.gait_guard_enabled
                              else bool(self._should_eval(step_offset + step)))
            if evaluation_due and self.eval_episode_num > 0 and self.eval_envs is not None:
                self._evaluate_and_checkpoint(agent, step_offset + step)
            if self._stop_requested:
                break
            # Periodic checkpoint -- previously only a single "latest.pt" was ever saved,
            # at the very end of the whole run (train.py), which discards the entire
            # behavioral history of a run and leaves no way to recover if it regresses
            # after this point.
            if self._should_checkpoint(step_offset + step):
                self.save_checkpoint(agent, f"step_{step_offset + step:07d}.pt", step=step_offset + step)
                self._rotate_periodic_checkpoints()
            # Live pose injection (see _check_pose_injection docstring). Blocks
            # this thread while waiting, same as the eval/checkpoint calls
            # above -- training only ever pauses at this clean loop boundary.
            if self._check_pose_injection(agent, step_offset + step):
                injection_report_steps = 50
                agent_state = agent.get_initial_state(envs.env_num)
                act = agent_state["prev_action"].clone()
                done = torch.ones_like(done)
                returns.zero_()
                lengths.zero_()
                video_cache = []
            # Save metrics
            # One reset-mask transfer replaces a GPU synchronization per world.
            # Keep per-episode records and absolute step accounting unchanged.
            done_host = done.detach().cpu().numpy()
            if done_host.any():
                episode_values = torch.stack((returns, lengths.to(returns.dtype)), dim=1).detach().cpu().numpy()
                for i in np.flatnonzero(done_host):
                    if episode_values[i, 1] > 0:
                        if i == 0 and len(video_cache) > 0:
                            video = torch.stack(video_cache, axis=0)
                            self.logger.video("train_video", tools.to_np(video[None]))
                            video_cache = []
                        self.logger.scalar("episode/score", episode_values[i, 0])
                        self.logger.scalar("episode/length", episode_values[i, 1])
                        self._write_log(agent, step_offset + step + int(i))  # distinct tensorboard points
                returns.masked_fill_(done, 0.)
                lengths.masked_fill_(done, 0)
            step += int((~done_host).sum()) * self._action_repeat  # step is based on env side
            lengths += ~done

            # Step environments.  Each env backend handles device placement
            # internally (ParallelEnv converts to CPU, IsaacLabVecEnv keeps
            # on GPU).  The .to() calls below are no-ops when the data is
            # already on agent.device.
            # (B, A), (B,)
            with self._timed_phase("environment", agent):
                trans, step_done = envs.step(act.detach(), done)
            if hasattr(envs, "write_live_view") and time.perf_counter() - last_live_view > .1:
                # ~10 Hz snapshot for the live viewer; one tiny GPU->host copy.
                envs.write_live_view(self.logdir / "live_view.npz", trans["reward"][:, 0])
                last_live_view = time.perf_counter()
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            # (B,)
            done = step_done.to(agent.device)
            if injection_report_steps > 0:
                # Real GPU-sim reward of the injected worlds, so a pose's effect is visible.
                injection_report_steps -= 1
                if injection_report_steps % 10 == 0:
                    print(f"  post-injection t={50 - injection_report_steps:2d}: reward/world "
                          f"{[round(r, 1) for r in trans['reward'][:, 0].tolist()]}  fell="
                          f"{trans['is_terminal'][:, 0].int().tolist()}", flush=True)

            # Policy inference on GPU.
            # "agent_state" is reset by the agent based on the "is_first" flag in trans.
            # (B, A)
            with self._timed_phase("policy", agent):
                act, agent_state = agent.act(trans.clone(), agent_state, eval=False)

            # Store transition.
            # We keep the observation and the action that produced it together.
            # Mask actions after an episode has ended.
            trans["action"] = act * ~done.unsqueeze(-1)
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            trans["episode"] = episode_ids  # Don't lift dim
            if "image" in trans:
                if self.train_video:
                    video_cache.append(trans["image"][0])
            with self._timed_phase("replay_write", agent):
                self.replay_buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]
            # Update models after enough data has accumulated
            if self.replay_buffer.count() // envs.env_num > self.batch_length + 1:
                if self._should_pretrain():
                    update_num = self.pretrain
                else:
                    update_num = self._updates_needed(step)
                encoder_phase_changed = False
                for _ in range(update_num):
                    if self._stop_requested:
                        break
                    verify_now = self.require_learning and not first_update_verified
                    before = ({name: p.detach().clone() for name, p in agent.named_parameters()
                               if p.requires_grad} if verify_now else None)
                    with self._timed_phase("optimizer", agent):
                        if self.imitation_dataset is None:
                            _metrics = agent.update(self.replay_buffer)
                        else:
                            settings = self.imitation_settings
                            expert_batch = self.imitation_dataset.sample(
                                int(settings.batch_size), int(settings.batch_length),
                                burnin=int(settings.burn_in), split="train", device=agent.device,
                                rng=self.imitation_rng)
                            _metrics = agent.update(self.replay_buffer, expert_batch=expert_batch)
                    if verify_now:
                        changed = sum(not torch.equal(before[name], p.detach())
                                      for name, p in agent.named_parameters() if name in before)
                        finite = all(bool(torch.isfinite(p).all()) for p in agent.parameters())
                        if not changed or not finite:
                            raise RuntimeError("First optimizer update failed: unchanged or nonfinite weights")
                        first_update_verified = True
                        print(f"LEARNING VERIFIED: {changed} parameter tensors changed; "
                              f"step={step_offset + step}; optimizer_updates={self.update_count + 1}", flush=True)
                        del before
                    train_metrics = _metrics
                    self.update_count += 1
                    if (self.gait_guard_enabled and not self._stop_requested and
                            self.update_count - self._gait_last_eval_updates >= self.gait_guard_eval_updates):
                        self._evaluate_and_checkpoint(agent, step_offset + step)
                    if (not self._stop_requested and self.stop_after_optimizer_updates and
                            self.update_count - initial_updates >= self.stop_after_optimizer_updates):
                        self.request_stop("optimizer_update_limit")
                    if self.encoder_training_aid and self.update_count in self.encoder_boundaries:
                        encoder_phase_changed = True
                        break  # never optimize past a phase boundary on old replay
                # Also drives non-observation schedules such as gradual action
                # authority. Sending once per update batch avoids per-gradient IPC.
                envs.set_training_updates(self.update_count)
                # Log training metrics
                if self._should_log(step_offset + step):
                    for name, value in train_metrics.items():
                        value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                        self.logger.scalar(f"train/{name}", value)
                    self.logger.scalar("train/opt/updates", self.update_count)
                    self.logger.scalar("train/opt/updates_this_run", self.update_count - initial_updates)
                    self._log_performance(step_offset + step)
                    if self.video_pred_log:
                        data, _, initial = self.replay_buffer.sample()
                        self.logger.video("open_loop", tools.to_np(agent.video_pred(data, initial)))
                    if self.params_hist_log:
                        for name, param in agent._named_params.items():
                            self.logger.histogram(name, tools.to_np(param))
                    self._write_log(agent, step_offset + step, fps=True)
                if encoder_phase_changed:
                    # Both boundaries start fresh episodes/replay, preserving learned
                    # weights and optimizer but not privileged observation history.
                    envs.set_training_updates(self.update_count)
                    self.replay_buffer.clear()
                    agent_state = agent.get_initial_state(envs.env_num)
                    act = agent_state["prev_action"].clone()
                    done = torch.ones_like(done)
                    returns.zero_()
                    lengths.zero_()
                    video_cache = []
                    print(f"Encoder phase changed at optimizer update {self.update_count}; "
                          "replay and recurrent state cleared; collecting fresh observations.")
            if (self.require_learning and self.learning_start_deadline_steps > 0
                    and step >= self.learning_start_deadline_steps
                    and self.update_count == initial_updates and not self._stop_requested):
                raise RuntimeError("Learning deadline reached without any optimizer updates")
            # The counters now describe completed collection and optimizer work.
            # SIGINT/SIGTERM only set a flag; they never serialize partial updates.
            self.last_train_step = int(step_offset + step)
        self.last_train_step = int(step_offset + step)
        if self.require_learning and self.update_count == initial_updates and not self._stop_requested:
            raise RuntimeError("Budget ended with zero optimizer updates; this was collection, not training")
        if self._stop_requested:
            self.save_checkpoint(agent, "latest.pt", step=self.last_train_step)
            print(f"Stopped at completed boundary: step={self.last_train_step}; "
                  f"optimizer_updates={self.update_count}; reason={self._stop_reason}; "
                  "saved latest.pt (replay is not checkpointed)", flush=True)
        # Evaluate the final weights, even when the budget ends between eval ticks.
        if (self.eval_episode_num > 0 and self.eval_envs is not None and
                (not self._stop_requested or self._stop_reason == "optimizer_update_limit") and
                (not self.gait_guard_enabled or self._gait_last_eval_updates != self.update_count)):
            self._evaluate_and_checkpoint(agent, self.last_train_step)
        self._log_performance(self.last_train_step)
        self._write_log(agent, self.last_train_step)
