# r2dreamer humanoid walking: project notes (state on 12 Oct 2026)

## Goal
Make the 19-joint servo humanoid walk (dots 5 cm ahead of each foot's toe, then 100 m straight) in MuJoCo/MJX, trained with a
DreamerV3-style learner (r2dreamer), then move to the real robot (STM32N6 + 2x PCA9685 + MG996R-class servos).

## Main findings
1. The vendor gait (validation/vendor_forward_3_steps.csv) CANNOT step in this simulator (falls at ~5 s, feet never leave the floor),
   so every earlier training run (up to 14M steps) never produced a step.
2. scripts/scripted_walk.py: quasi-static inverse-kinematics walk. Works (60 s) but slow (0.37 cm/s) and waddles.
3. scripts/humanlike_walk.py (+ scripts/zip_humanlike_gait.py = ZMP preview-control planner from the user's files.zip, unchanged):
   human-like walk using OUR 5-joint-leg IK. Best settings (BEST in the file): 0.40 s/step, 2.5 cm half-stride (5 cm per foot
   per step), zmp_inset 0.02, toe_pitch 0, tilt stabiliser (roll -> ankle+hip roll; k=1.0, d=0.04, hip sign -1, ankle signs mirrored),
   arm swing 0.6. Result in CPU sim: 84 s, 6.2 cm/s, straight (heading +2.5 deg), tilt < 3.2 deg.
4. Hardware (user's tests, bundle robot_hardware_complete_20261012): torso IMU axes are FLU = (-raw X, -raw Y, +raw Z) (the model
   had (-Y, +X, +Z)); IMU logs at 40-50 Hz; knees are MG996R at 6 V = ~1.08 N*m (sim assumes 1.77 N*m on all joints!);
   no servo feedback; firmware has no 50 Hz streaming command path.
5. Sim realism tests of the walk: at 1.08 N*m it still walks but shaky; at 0.9 N*m falls at 11 s; delay <= 40 ms ok, 60 ms falls;
   all real limits combined (1.08 N*m, 5.5 rad/s, 0.6 deg resolution, 40 ms delay, IMU noise) -> falls at 4.6 s.
   => the walk does NOT yet work under real servo limits. Torque (hip roll) is the main problem.
6. RL status: a learner on top of the reference (small corrections +-5.7 deg) kept the walk in some runs (local 64-robot test:
   score 2121 -> 5049, drift penalty 1204 -> 19) but degraded it in a 512-robot / train_ratio 64 cloud run. Not conclusive.
   The learner has NEVER walked without the reference.

## Layout (this folder = ~/r2dreamer)
- envs/mjx_walking/env.py        the environment (rewards goal_corridor_v3, dots, stall rules, stabiliser, phase pool, clock control)
- envs/mjx_walking/scene.py      scene build, gait-reference loaders, pose pools
- envs/mjx_walking/assets/       robot scenes (humanoid_rubber_2mm_servo_v2_kp40.xml = default), meshes, humanlike_gait.npz,
                                 humanlike_phase_pool.npz, scripted_gait.npz, manual_rsi_poses_merged.json
- envs/mjx_teacher_env.py        wraps the env for training (config keys mjx_*)
- scripts/humanlike_walk.py      human-like walk: --view (viewer), --video out.mp4 (needs MUJOCO_GL=glfw), --export-training A B
- scripts/scripted_walk.py       slow scripted walk + shared kinematics (Kin) and CPU playback
- scripts/interactive_rsi.py     pose editor with live injection (keys: P pause, I inject, F foot pin, G group, B balance, U undo, L lock)
- scripts/rsi_bridge.py          sync the editor with a run on a remote machine
- scripts/check_goal_corridor_reward.py   85 reward sanity tests (must print 85/85)
- scripts/deploy/vast_setup.sh, vast_train.sh   cloud setup + training launcher (STAGE=walk ... see below)
- docs/                          earlier write-ups

## Commands
Setup on a new machine (Linux/WSL, NVIDIA driver with CUDA >= 12.8):  bash scripts/deploy/vast_setup.sh   (builds .venv from requirements-lock.txt)
Reward tests:   .venv/bin/python scripts/check_goal_corridor_reward.py | tail -1        # 85/85
Watch the walk: .venv/bin/python scripts/humanlike_walk.py --view
Record video:   MUJOCO_GL=glfw .venv/bin/python scripts/humanlike_walk.py --video out.mp4      (EGL crashes on the laptop; glfw works)
Train (cloud 4090): STAGE=walk ENV_NUM=64 WALK_STEPS=3000000 EVAL_EVERY=50000 TRAIN_RATIO=512 BUFFER=262144 VIDEO=1 FROM_CKPT=null \
                    LOGDIR_ROOT=/workspace/runs bash scripts/deploy/vast_train.sh        (defaults = human-like walk, std 0.005-0.03, scale 0.1)
Stop safely:    kill -INT <train.py pid>   (saves latest.pt); same command again resumes.

## Traps (learned the hard way)
- pkill -f matches its own shell: use pgrep -f "^.venv/bin/python -u train.py" and kill by pid.
- Never sed with # as delimiter on code containing '#'.
- "steps" in the logs count 10 physics steps per robot decision; 1M steps at 512 robots is only ~2 s of life per robot.
- Training must start at random walk phases (phase pool) or the learner never sees later phases; evaluation starts standing.
- University wifi blocked SSH to vast.ai; use a phone hotspot.
- Derived scenes (*_mjx_contacts_*.xml) are generated and cached; delete them after editing a scene.
- Real servo neutral/degree/limits are not calibrated; default firmware bounds 1200-1800 us (about +-36 deg, unverified)
  are smaller than the walk needs (hip/thigh +-46-51 deg).

## Next steps
1. Per-joint servo models (only the knees are confirmed MG996R). 2. Retrain/validate with real limits: per-joint torque, 5-7 rad/s,
20-50 ms delay, IMU model (measured axes, 40-50 Hz, noise), 0.6 deg resolution, mass 2.66 kg, pushes + randomisation.
3. Reduce hip-roll torque demand (narrower stance, lower CoM, shorter steps). 4. Measure: loaded supply voltage/current, mass,
command latency, deg per us. 5. Firmware: 50 Hz streaming of 19 targets, stop command, gait table + tilt feedback on the STM32.
6. Optional: distil reference+learner into a standalone policy.
