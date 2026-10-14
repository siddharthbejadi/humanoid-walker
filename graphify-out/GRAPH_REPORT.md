# Graph Report - humanoid-walker  (2026-10-14)

## Corpus Check
- 178 files · ~269,441 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 59 file(s) not represented in the graph (top: .stl 47, .xml 7, (none) 1)

## Summary
- 1831 nodes · 4117 edges · 115 communities (74 shown, 41 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 221 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Demonstration Audit
- Student Policy Runtime
- Imitation Checkpointing
- Reference Reset Tests
- Interactive RSI Design
- Community 5
- Checkpoint Inspection & Packaging
- Audit Expert & Provider
- MJX Teacher Env
- Script-Free Contract & Imports
- Script-Free Contract & Imports
- Community 11
- Dreamer Agent
- Demonstration Audit
- Scene & Pose Pools
- Walking Env & Rewards
- Imitation Training
- Teacher Curriculum Supervisor
- Community 18
- Human-like Walk Script
- Imitation Training
- Script-Free Runbook
- Scene & Pose Pools
- Reference Preflight
- MJX Walking Env Core
- Community 25
- Checkpoint Ranking
- Package Tests
- Capacity Video Validation
- Step Clearance Tests
- ZMP Gait Planner
- Checkpoint Ranking
- Project Notes & Hardware
- Community 33
- RSSM Dynamics
- Community 35
- Step Clearance Tests
- Trainer & Evaluation Tests
- Community 38
- Community 39
- Community 40
- Community 41
- Script-Free Runbook
- Watch Policy Viewer
- Community 44
- Walking Env & Rewards
- Community 46
- Checkpoint Ranking
- Trainer & Evaluation Tests
- MJX Teacher Env Bridge
- Teacher Session Plan
- Community 51
- Community 52
- Community 53
- Community 54
- Community 55
- Dreamer Agent
- Community 57
- Community 58
- Script-Free Runbook
- Community 60
- Community 61
- MJX Teacher Env Bridge
- Teacher Session Plan
- Community 64
- Trainer & Evaluation Tests
- Community 66
- Community 67
- Community 68
- Script-Free Runbook
- Community 70
- Community 71
- Community 72
- Teacher Session Plan
- Community 74
- Checkpoint Ranking
- Community 76
- Community 77
- Script-Free Runbook
- Community 79
- Community 80
- Community 81
- Community 82
- Community 83
- Community 84
- Community 85
- Community 86
- Teacher Session Plan
- Community 88
- Community 89
- Checkpoint Ranking
- Community 91
- Imitation Checkpointing
- Community 93
- Community 94
- Community 95
- Community 96
- Community 97
- Community 98
- Community 99
- Checkpoint Ranking
- Community 101
- Checkpoint Ranking
- Trainer & Evaluation Tests
- Community 104
- Community 105
- Community 106
- Community 107
- Community 108
- Community 110
- Community 111
- Community 112
- Community 113
- Community 114

## God Nodes (most connected - your core abstractions)
1. `MJXTeacherEnv` - 48 edges
2. `Dreamer` - 42 edges
3. `OnlineTrainer` - 35 edges
4. `build_training_contract()` - 30 edges
5. `check_checkpoint_contract()` - 29 edges
6. `ImitationDataset` - 27 edges
7. `PipelineTests` - 25 edges
8. `main()` - 25 edges
9. `Project notes 12 Oct 2026` - 25 edges
10. `ScriptFreeContractTests` - 24 edges

## Surprising Connections (you probably didn't know these)
- `Closed-loop results by variant (5k/12k updates, slew 6/8)` --semantically_similar_to--> `Closed loop 1.32-1.40 m forward, 39-41 valid steps, 0/0/3% falls`  [INFERRED] [semantically similar]
  docs/PROJECT_HISTORY.md → README.md
- `Sim torque 1.77 N m vs real knee ~1.08 N m; gait falls at 1.08 in sim` --semantically_similar_to--> `Action range is stability knob: +-1.0 rad collapses, +-22.5 deg works`  [INFERRED] [semantically similar]
  deploy/DEPLOYMENT_SPEC.md → r2dreamer/docs/PROJECT_REVIEW_20261009.md
- `run()` --uses--> `ScriptFreePolicy`  [INFERRED]
  deploy/robot_host.py → r2dreamer/script_free_policy.py
- `main()` --uses--> `ScriptFreePolicy`  [INFERRED]
  deploy/watch_policy.py → r2dreamer/script_free_policy.py
- `main()` --uses--> `GPUStudentProvider`  [INFERRED]
  deploy/watch_policy.py → r2dreamer/scripts/collect_script_free_demonstrations.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Staged hardware test gates** — deploy_robot_test_protocol_gate0, deploy_robot_test_protocol_gate1, deploy_robot_test_protocol_gate2, deploy_robot_test_protocol_gate3, deploy_robot_test_protocol_gate4 [EXTRACTED 1.00]
- **Script-free BC-then-RL pipeline** — r2dreamer_docs_script_free_training_20261013_v2_expert_collector, r2dreamer_docs_script_free_training_20261013_v2_bc_first, r2dreamer_docs_script_free_training_20261013_v2_bounded_pilot, r2dreamer_docs_script_free_training_20261013_v2_checkpoint_eval [EXTRACTED 1.00]
- **Limp servo root cause to fix** — r2dreamer_docs_walking_fixes_20261008_candidate_scene_kp, r2dreamer_docs_walking_fixes_20261008_stand_test, r2dreamer_docs_walking_fixes_20261008_servo_scene [EXTRACTED 1.00]
- **Behaviour-cloning pipeline** — r2dreamer_scripts_collect_script_free_demonstrations, r2dreamer_scripts_pretrain_script_free, r2dreamer_scripts_evaluate_script_free, deploy_robot_host [EXTRACTED 1.00]
- **Dead ends** — rl_fine_tuning_dead_end, dart_dead_end, startup_fraction_05_dead_end, vendor_gait_dead_end [EXTRACTED 1.00]

## Communities (115 total, 41 thin omitted)

### Community 0 - "Demonstration Audit"
Cohesion: 0.06
Nodes (16): validate_trajectory_contract(), align_episode(), file_sha256(), ImitationDataset, save_dataset(), split_statistics(), validate_arrays(), collect_batch() (+8 more)

### Community 1 - "Student Policy Runtime"
Cohesion: 0.05
Nodes (20): pose(), validate_student_environment(), _devices(), ScriptFreePolicy, agent(), data(), ImitationLearnerTests, record() (+12 more)

### Community 2 - "Imitation Checkpointing"
Cohesion: 0.10
Nodes (4): normalize_motor_targets(), full_command_contract(), _full_environment(), config()

### Community 3 - "Reference Reset Tests"
Cohesion: 0.06
Nodes (11): motor_targets_from_actions(), FullCommandTests, method(), namespace(), reward(), fixture(), load_source(), ReferenceResetTests (+3 more)

### Community 4 - "Interactive RSI Design"
Cohesion: 0.06
Nodes (35): Interactive RSI design brainstorm, Backup around every injection (checkpoint, pre/post state, jsonl log), Open question: CPU preview vs MJX parity, Damped-least-squares IK with opposite foot pinned, Height-sag reward bug: no gradient before -10cm cliff, Hierarchy-aware manipulation: FK head, IK feet, springy torso, Interactive RSI / Director Mode, Live injection into running training, side-channel handoff (+27 more)

### Community 5 - "Community 5"
Cohesion: 0.05
Nodes (37): Sim torque 1.77 N m vs real knee ~1.08 N m; gait falls at 1.08 in sim, Project review why walking stalled, Action range is stability knob: +-1.0 rad collapses, +-22.5 deg works, Die-early optimum fixed with alive bonus 0.5/step, Exploration std 0.01-0.06; large noise crashed into back wall, goal_corridor_v3 reward profile, Root cause: simulated servos too soft to hold robot, Old soft-servo checkpoints carry wrong physics (+29 more)

### Community 6 - "Checkpoint Inspection & Packaging"
Cohesion: 0.11
Nodes (16): Graceful stop, status JSON, hashed backup; not auto-resume, main(), main(), main(), main(), source_files(), main(), source_files() (+8 more)

### Community 7 - "Audit Expert & Provider"
Cohesion: 0.06
Nodes (5): AuditExpert, evaluation_row(), PipelineTests, stage(), unsafe_rollouts()

### Community 8 - "MJX Teacher Env"
Cohesion: 0.08
Nodes (5): make_envs(), MJXTeacherEnv, main(), main(), main()

### Community 9 - "Script-Free Contract & Imports"
Cohesion: 0.11
Nodes (17): check_imitation_source(), validate_student_config(), input_snapshot(), main(), read_json(), recovery_command(), source_snapshot(), validate_command() (+9 more)

### Community 10 - "Script-Free Contract & Imports"
Cohesion: 0.12
Nodes (23): ScriptedExpert, checkpoint_report(), compose(), digest(), evaluate(), evaluation_rows(), execute_pipeline(), finite_json() (+15 more)

### Community 11 - "Community 11"
Cohesion: 0.08
Nodes (9): BlockLinear, LambdaLayer, MLP, MLPHead, MultiEncoder, Projector, ReturnEMA, Deter (+1 more)

### Community 12 - "Dreamer Agent"
Cohesion: 0.12
Nodes (3): Dreamer, _agc(), factory()

### Community 13 - "Demonstration Audit"
Cohesion: 0.11
Nodes (4): main(), main(), main(), main()

### Community 14 - "Scene & Pose Pools"
Cohesion: 0.10
Nodes (9): build_mjx_scene(), validate_teacher_state(), _measure(), validate_reset_repeatability(), check_schedule(), main(), targets(), main() (+1 more)

### Community 15 - "Walking Env & Rewards"
Cohesion: 0.11
Nodes (12): alternating_step_events(), autonomous_walk_reward_terms(), foot_landing_events(), foot_swing_qualification(), footstep_target_terms(), distance(), target(), goal_corridor_reward_terms() (+4 more)

### Community 16 - "Imitation Training"
Cohesion: 0.13
Nodes (6): imitation_objective(), masked_mean(), sequence_mask(), barlow_objective(), sampled_effective_rank(), clip_grad_agc_()

### Community 17 - "Teacher Curriculum Supervisor"
Cohesion: 0.15
Nodes (12): main(), checkpoint_metadata(), main(), phase_overrides(), phases(), validate_phase_result(), write_report(), case_overrides() (+4 more)

### Community 18 - "Community 18"
Cohesion: 0.08
Nodes (11): compute_global_norm(), compute_rms(), convert(), CudaBenchmark, Every, Once, print_module_tree(), print_param_stats() (+3 more)

### Community 19 - "Human-like Walk Script"
Cohesion: 0.15
Nodes (12): arm_signal(), build(), export(), main(), play(), render_video(), on_step(), robot_geom() (+4 more)

### Community 20 - "Imitation Training"
Cohesion: 0.10
Nodes (6): full_episode_imitation_evidence(), episodes(), ImitationReportingTests, evidence(), MemoryAgent, MemoryRSSM

### Community 21 - "Script-Free Runbook"
Cohesion: 0.17
Nodes (4): make_environment(), TrainingContractTests, build_training_contract(), check_checkpoint_contract()

### Community 22 - "Scene & Pose Pools"
Cohesion: 0.13
Nodes (8): build_pose_rsi_pool(), build_scripted_rsi_pool(), build_vendor_rsi_pool(), rsi_injection_flags(), scripted_gait_reference(), sha256(), vendor_gait_reference(), validate_script_free_settings()

### Community 23 - "Reference Preflight"
Cohesion: 0.16
Nodes (13): asset(), capacity_probe(), compose(), digest(), gpu_baseline(), gpu_runtime_report(), gpu_worker(), main() (+5 more)

### Community 25 - "Community 25"
Cohesion: 0.10
Nodes (5): Conv2dSamePad, ConvDecoder, ConvEncoder, MultiDecoder, RMSNorm2D

### Community 27 - "Package Tests"
Cohesion: 0.20
Nodes (9): Earlier-experiment scripts kept because manifest/tests reference them, ScriptFreePackageTests, source_fixture(), generated_contact_scene(), reject_unlisted_source(), runtime_path(), sha256_file(), verify() (+1 more)

### Community 28 - "Capacity Video Validation"
Cohesion: 0.17
Nodes (3): validate_capacity(), validate_capacity_videos(), ReferencePreparationTests

### Community 29 - "Step Clearance Tests"
Cohesion: 0.14
Nodes (10): build_path(), emit(), export_phase_pool(), export_table(), load(), main(), simulate(), smooth() (+2 more)

### Community 30 - "ZMP Gait Planner"
Cohesion: 0.19
Nodes (12): export_csv(), GaitParams, generate(), leg_fk(), leg_ik(), min_jerk(), plan_footsteps(), preview_control() (+4 more)

### Community 31 - "Checkpoint Ranking"
Cohesion: 0.21
Nodes (5): checkpoint_rank(), finish_trial_success(), walking_trial_success(), load_class(), validate_learning_budget()

### Community 32 - "Project Notes & Hardware"
Cohesion: 0.14
Nodes (17): Project notes 12 Oct 2026, Goal: 19-joint humanoid walks 100 m in MuJoCo/MJX, then real robot, Hardware: IMU FLU axes, 40-50 Hz, knee MG996R ~1.08 N m, no servo feedback, Hardware: STM32N6 + 2x PCA9685 + MG996R servos, Human-like ZMP gait: 0.40 s/step, 6.2 cm/s, tilt<3.2 deg, Next steps: servo models, real-limit retraining, firmware 50 Hz streaming, distil, Random-phase pool needed so learner sees later gait phases, RL on reference inconclusive; learner never walked without reference (+9 more)

### Community 33 - "Community 33"
Cohesion: 0.16
Nodes (6): Buffer, check_previous_actions_and_observations_keep_exact_causal_alignment(), check_recomputed_latents_update_only_the_sampled_storage_indices(), CPUReplayActionAlignmentTests, CUDAReplayActionAlignmentTests, populated_buffer()

### Community 36 - "Step Clearance Tests"
Cohesion: 0.24
Nodes (4): count_landings(), qualify(), step_rows(), StepClearanceTests

### Community 37 - "Trainer & Evaluation Tests"
Cohesion: 0.14
Nodes (4): CurriculumTests, Agent, Logger, Replay

### Community 38 - "Community 38"
Cohesion: 0.13
Nodes (14): BC error 1.7e-2 to 2.9e-6 held-out action MSE, Board STM32N6570-DK, ST-LINK COM port, 2x PCA9685 at 50 Hz, Checkpoints README, One NVIDIA GPU (RTX 5090), CUDA >= 12.8, 19-servo humanoid (STM32N6 + 2x PCA9685 + MG996R-class servos), graphify knowledge graph: 1750 nodes, 3988 edges, 118 communities, Work on Linux filesystem copy, not Windows mount, MuJoCo 3.14 + MJX-Warp GPU simulator, 20 ms control step, 10 sub-steps (+6 more)

### Community 39 - "Community 39"
Cohesion: 0.21
Nodes (5): crc16_ccitt(), decode_frame(), encode_frame(), _selftest(), SerialServoLink

### Community 41 - "Community 41"
Cohesion: 0.20
Nodes (8): build_observation(), head_frame(), main(), raw_to_sim_frame(), rotation_between(), run(), tilt_degrees(), Robot host CPU ~7-8 ms per step

### Community 42 - "Script-Free Runbook"
Cohesion: 0.13
Nodes (13): Dead end: DART noisy-action demonstrations (0-75% walking), Project history: what was tried, what failed, what is left, Preflight-only default; capacity route with 24 episodes, JAX_PLATFORM_NAME, JAX_PLATFORMS, MUJOCO_GL, PYTHONUNBUFFERED, start_script_free.sh script (+5 more)

### Community 43 - "Watch Policy Viewer"
Cohesion: 0.17
Nodes (4): check_host(), main(), check_probe(), main()

### Community 44 - "Community 44"
Cohesion: 0.13
Nodes (14): Scripted expert falls under measured servo limits (4.6 s), Gait search: 37 valid steps at 18 kg cm, only 2 under measured limits, NEXT: Gates 0-4 physical walk test protocol, Resting IMU tilt 14-16 deg vs sim; stand-still IMU calibration, Masked sensors: foot pressure (8) and magnetometer (3), No hardware test; servo calibration, signs, IMU axes unverified, Pipeline status DONE / NOW / NEXT / LATER, Realistic = sensing and latency (40 ms delay, ICM-20948 noise, 6 rad/s slew, masked sensors), not torque (+6 more)

### Community 45 - "Walking Env & Rewards"
Cohesion: 0.19
Nodes (11): foot_progress_records(), walking_reward_terms(), reward_only_report(), inputs(), landing_inputs(), landing_record_trajectory(), one_second(), progress_inputs() (+3 more)

### Community 46 - "Community 46"
Cohesion: 0.29
Nodes (5): check(), forbidden_reference(), main(), GPUStudentProvider, main()

### Community 47 - "Checkpoint Ranking"
Cohesion: 0.28
Nodes (4): Agent, EvaluationRecordTests, ResumeSelectionTests, trainer()

### Community 49 - "MJX Teacher Env Bridge"
Cohesion: 0.15
Nodes (5): --inject-rundir live injection (pause flag, key P / I), _observation(), select_reset(), balanced_pose_reset_indices(), grouped_pose_reset_indices()

### Community 50 - "Teacher Session Plan"
Cohesion: 0.16
Nodes (14): Next teacher session plan, Capacity probe: 64/128/256 worlds, 32 updates, Cooperative SIGINT/SIGTERM checkpoint, atomic latest.pt, If both arms fail: build validated balance/transfer/lift reference, Fixed balanced evaluation schedule (standing + each pose, 4 trials), FPS units: ~281 ticks/s = ~28 control transitions/s (action_repeat 10), GPU scope: JAX MJX-Warp physics, CUDA Torch learner, DLPack bridge, Single ablation: replay ratio 16 vs 64 (+6 more)

### Community 51 - "Community 51"
Cohesion: 0.21
Nodes (13): Script-free student deployment spec, target_rad = (pi/2)*action, clip, 6 rad/s slew limit, Behaviour cloning only, no RL at runtime, contract_clean12k.json exact contract, IMU frame remap sim=(-rawY,+rawX,+rawZ), qualitative, 45-channel input layout (cmd, IMU, mag, head, const, foot pressure), 19-joint order (legs, arms, head, hand tips), Magnetometer and foot pressure zeroed (masked in training) (+5 more)

### Community 52 - "Community 52"
Cohesion: 0.27
Nodes (3): gait_metrics(), preservation_result(), PreservationTests

### Community 58 - "Community 58"
Cohesion: 0.23
Nodes (5): kl(), symexp(), symlog(), symlog_mse(), SymlogDist

### Community 59 - "Script-Free Runbook"
Cohesion: 0.21
Nodes (10): Script-free training V2 runbook, BC-first then R2Dreamer online RL with supervised command loss; no DAgger, Bounded pilot: 64 dense episodes, 1000 BC updates, 1M tick online, Checkpoint-only evaluation with scripted loaders poisoned, Checkpoint binds data_sha256 and manifest_sha256, External expert collector: humanlike reference + roll stabilizer, Safe-label gating: failed expert rollouts get no BC labels, Whole seed-group split 48/8/8 train/val/test (+2 more)

### Community 60 - "Community 60"
Cohesion: 0.17
Nodes (3): ConsoleLogHandle, setup_console_log(), Tee

### Community 61 - "Community 61"
Cohesion: 0.18
Nodes (3): Imu, SerialImu, SyntheticImu

### Community 62 - "MJX Teacher Env Bridge"
Cohesion: 0.29
Nodes (7): compare_gradients(), diagnose(), fixed_batch(), gradient_groups(), main(), on_device(), parameter_digest()

### Community 63 - "Teacher Session Plan"
Cohesion: 0.24
Nodes (11): Declared success/failure and stop rules before looking at rewards, Teacher training evidence audit, 14M run: 88/88 falls, zero valid steps, Hypotheses: command semantics, unqualified resets, slew saturation, Checkpoints with SHA-256 (12.0M, 14.0M, best), Fresh 200k pilot: 40/40 falls, Survival 0.90 -> 0.48 s, slew-limited 72-86%, tilt rising, Saved poses fall in 0.60-0.96 s at 18 kg-cm on CPU (+3 more)

### Community 64 - "Community 64"
Cohesion: 0.29
Nodes (6): compare(), difference(), main(), alias_measurements(), host(), snapshot()

### Community 65 - "Trainer & Evaluation Tests"
Cohesion: 0.29
Nodes (3): EvaluationEnvs, TrainerGuardTests, evaluate()

### Community 69 - "Script-Free Runbook"
Cohesion: 0.24
Nodes (7): _array(), _configured_file(), _differences(), _hash_arrays(), _hash_file(), _scene_files(), visit()

### Community 70 - "Community 70"
Cohesion: 0.25
Nodes (4): Closed-loop results by variant (5k/12k updates, slew 6/8), Closed loop 1.32-1.40 m forward, 39-41 valid steps, 0/0/3% falls, save(), Seed-1100 rollout: 1.20 m, 33 valid steps (17 L / 16 R)

### Community 71 - "Community 71"
Cohesion: 0.36
Nodes (9): 40 ms (2-step) command delay in training, Staged robot test protocol, Gate 0 bench, no policy, Gate 1 plumbing, servos disabled (dry-run), Gate 2 hanging, torque limited, Gate 3 harness on ground, Gate 4 free walk 1 m course, ServoLink.relax() torque cut, limp within 100 ms (+1 more)

### Community 72 - "Community 72"
Cohesion: 0.28
Nodes (5): binary(), symexp_twohot(), TwoHot, to_f32(), to_i32()

### Community 73 - "Teacher Session Plan"
Cohesion: 0.22
Nodes (8): Pilot budget 200k counted ticks per arm, 64 worlds, JAX_PLATFORM_NAME, JAX_PLATFORMS, PYTHONPATH, start_teacher_next.sh script, TEACHER_RUN_DIR, TEACHER_SCENE, XLA_PYTHON_CLIENT_PREALLOCATE

### Community 75 - "Checkpoint Ranking"
Cohesion: 0.25
Nodes (5): enable_deterministic_run(), recursively_load_optim_state_dict(), _apply_action_scale_compensation(), main(), _transfer_world_model()

### Community 76 - "Community 76"
Cohesion: 0.25
Nodes (8): Observed result: alternating left/right steps with arm swing, Contact sheet of humanoid walking frames (student_realistic_12k, seed 1100), Frame 1: robot upright, feet together, arms at sides (start), Frame 2: first step, arms swinging outward, Frame 3: mid-walk, opposite leg and arm phase, Frame 4: later frame, robot seen from side, legs staggered, MuJoCo simulation of 19-joint servo humanoid, 20 s, 1.2 m forward, Observed result: upright posture maintained, no fall

### Community 77 - "Community 77"
Cohesion: 0.32
Nodes (8): Reference-assisted policy pilot, Reference+stabilizer success must not be credited to learning, Hash-verified source ZIP and manifest, Pinned runtime: Torch 2.8, JAX 0.10.2, MuJoCo 3.14, Warp 1.17, Py3.11, Promotion: standing-start, ordered L/R supported landings, no falls, video, Readiness: structural, GPU reference diagnostics, 32-update capacity, Capacity admission needs 32 final-step MP4s with sidecars, Compare against zero-residual reference+stabilizer baseline

### Community 78 - "Script-Free Runbook"
Cohesion: 0.39
Nodes (8): Valid-step clearance change, Same trajectory: 15 mm gives right landings 7, left 0; 20 mm gives 0, training_contract.task.valid_step_clearance_m guards resume, Dots reward distinct from valid supported steps, Landing: earlier swing, hysteretic contact, >=20 mm forward progress, Left landings fail opposite-foot load 0.377-0.538 < 0.55, Valid-step minimum sole clearance 15 mm (inclusive >=), Swing predicate: load<0.10 BW, opposite load>0.55, stable torso, foot speed<0.03

### Community 79 - "Community 79"
Cohesion: 0.25
Nodes (7): JAX_PLATFORM_NAME, PYTHONPATH, start_mjx_teacher_pilot.sh script, TEACHER_RUN_DIR, TEACHER_SCENE, XLA_PYTHON_CLIENT_MEM_FRACTION, XLA_PYTHON_CLIENT_PREALLOCATE

### Community 80 - "Community 80"
Cohesion: 0.25
Nodes (7): JAX_PLATFORM_NAME, JAX_PLATFORMS, MUJOCO_GL, PYTHONPATH, start_reference_pilot.sh script, TEACHER_RUN_DIR, XLA_PYTHON_CLIENT_PREALLOCATE

### Community 87 - "Teacher Session Plan"
Cohesion: 0.38
Nodes (7): Sequential teacher curriculum, curriculum.json supervisor record, stops on any failure, Isolated-pose phases risk forgetting; mixed phase exposes it, init_from carries full agent state; transfer_from unsuitable, Final-eval videos per start with sidecars, GPU reset diagnostic: derived sensor tolerance bounds, Six-phase curriculum: standing, poses 1-4, mixed (8M ticks)

### Community 89 - "Community 89"
Cohesion: 0.29
Nodes (6): JAX_PLATFORM_NAME, PYTHONPATH, start_mjx_teacher_standing_pilot.sh script, TEACHER_RUN_DIR, TEACHER_SCENE, XLA_PYTHON_CLIENT_PREALLOCATE

### Community 90 - "Checkpoint Ranking"
Cohesion: 0.57
Nodes (3): bridge(), BridgeAccountingTests, state()

### Community 91 - "Community 91"
Cohesion: 0.33
Nodes (6): Hand-tip joint vs firmware gripper channel unconfirmed, Gait needs hip up to ~0.95 rad; firmware bounds ~+-36 deg too narrow, Mass 3.0 kg sim vs ~2.66 kg real, Open items: foot direction, neutrals, us/rad, latency, supply, true mass, Default servo centres (pulse widths) recorded by user, Supply sag, backlash, foot friction not modelled

### Community 92 - "Imitation Checkpointing"
Cohesion: 0.40
Nodes (3): 64 expert episodes x 1000 steps = 64 000 labelled steps, save_imitation_checkpoint(), main()

### Community 97 - "Community 97"
Cohesion: 0.33
Nodes (5): JAX_PLATFORM_NAME, PYTHONPATH, vast_setup.sh script, XLA_PYTHON_CLIENT_MEM_FRACTION, XLA_PYTHON_CLIENT_PREALLOCATE

### Community 98 - "Community 98"
Cohesion: 0.53
Nodes (4): check_cpu_standing(), check_gpu(), check_render(), fail()

### Community 101 - "Community 101"
Cohesion: 0.50
Nodes (3): Behaviour cloning of scripted gait (12 000 updates), Scripted human-like gait: ZMP preview control + roll stabiliser, Dead end: training on vendor gait (cannot step in simulator)

## Knowledge Gaps
- **116 isolated node(s):** `run9.sh script`, `STARTUP_FRACTION`, `setup_script_free.sh script`, `eval_many.sh script`, `probe.sh script` (+111 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 655 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **41 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `OnlineTrainer` connect `Checkpoint Ranking` to `Trainer & Evaluation Tests`, `Imitation Checkpointing`, `Trainer & Evaluation Tests`, `MJX Teacher Env`, `Script-Free Contract & Imports`, `Community 74`, `Checkpoint Ranking`, `Checkpoint Ranking`, `Trainer & Evaluation Tests`, `Community 57`, `Script-Free Runbook`, `Checkpoint Ranking`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **What connects `run9.sh script`, `STARTUP_FRACTION`, `setup_script_free.sh script` to the rest of the system?**
  _116 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Demonstration Audit` be split into smaller, more focused modules?**
  _Cohesion score 0.0579476861167002 - nodes in this community are weakly interconnected._
- **Why does `MJXTeacherEnv` connect `MJX Teacher Env` to `Community 64`, `Demonstration Audit`, `Community 98`, `Imitation Checkpointing`, `Checkpoint Inspection & Packaging`, `Community 40`, `Demonstration Audit`, `Scene & Pose Pools`, `Community 46`, `MJX Teacher Env Bridge`, `Reference Preflight`, `MJX Walking Env Core`, `Checkpoint Ranking`, `Script-Free Runbook`, `Imitation Checkpointing`, `Checkpoint Ranking`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Should `Student Policy Runtime` be split into smaller, more focused modules?**
  _Cohesion score 0.050724637681159424 - nodes in this community are weakly interconnected._
- **Why does `Walking fixes: servo root cause` connect `Community 5` to `MJX Teacher Env Bridge`, `Interactive RSI Design`, `Scene & Pose Pools`, `Walking Env & Rewards`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Should `Imitation Checkpointing` be split into smaller, more focused modules?**
  _Cohesion score 0.1012987012987013 - nodes in this community are weakly interconnected._