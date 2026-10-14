#!/usr/bin/env bash
# Two-stage servo training, from scratch (resumes a stage if its latest.pt exists).
#   STAGE=balance : standing reward, all poses mixed  -> runs/servo_balance
#   STAGE=walk    : goal_corridor_v3 walking, init from servo_balance/best.pt -> runs/servo_gait_v3
#   STAGE=both    : balance, then walk (default; run inside tmux)
# Usage:  STAGE=both ENV_NUM=1024 bash scripts/deploy/vast_train.sh
# Knobs:  ENV_NUM=64 BALANCE_STEPS=1000000 POSE_GROUPS=[0,1,2,3,4] POSTURE=1.0 BALANCE_RUN=servo_balance WALK_STEPS=20000000 TRAIN_RATIO=64 BUFFER=524288
#         EVAL_EVERY=1000000 BALANCE_SCALE=0.3927 WALK_SCALE=1.0 STEP_CLEARANCE_M=0.015 VIDEO=1 LOGDIR_ROOT=/workspace/runs FROM_CKPT=<override walk init>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
STAGE="${STAGE:-both}"
ROOT="${LOGDIR_ROOT:-/workspace/runs}"
if [ "$STAGE" = both ]; then
  STAGE=balance bash "$0"
  # A stopped (SIGINT) balance run also exits 0; only walk once it reached its budget.
  done_steps=$(.venv/bin/python -c "import torch; print(int(torch.load('$ROOT/${BALANCE_RUN:-servo_balance}/latest.pt', map_location='cpu', weights_only=False)['step']))")
  if [ "$done_steps" -lt "${BALANCE_STEPS:-1000000}" ]; then
    echo "Balance stopped early at step $done_steps; not starting walk. Re-run to resume balance."; exit 0
  fi
  STAGE=walk bash "$0"
  exit 0
fi
GL="$(cat .mujoco_gl 2>/dev/null || echo egl)"
VIDEO="${VIDEO:-1}"; [ "$GL" = none ] && VIDEO=0
export MUJOCO_GL="$GL"; [ "$GL" = osmesa ] && export PYOPENGL_PLATFORM=osmesa
EVERY="${EVAL_EVERY:-1000000}"
ENV_NUM="${ENV_NUM:-64}"
# Replay must hold batch_length+2 decisions per world before the first update;
# give the "no updates yet" guard 3x that (1024 worlds: ~690k counted steps).
DEADLINE=$(( ENV_NUM * 10 * 200 )); [ "$DEADLINE" -lt 100000 ] && DEADLINE=100000
# Joint range per stage. Balance uses the +-22.5 deg range that learned standing before: with
# kp-20 servos a 1.0 rad range turned small policy errors into full-torque shoves (the
# 2026-10-09 balance run fell backwards, 8/8 at 2M). Walk needs ~0.91 rad for the vendor
# gait; init_from_action_scale_compensation rescales the policy mean across the change.
EXTRA=()
case "$STAGE" in
  balance)
    LOGDIR="$ROOT/${BALANCE_RUN:-servo_balance}"; STEPS="${BALANCE_STEPS:-1000000}"; INIT="${FROM_CKPT:-null}"
    POSES=envs/mjx_walking/assets/manual_rsi_poses.json
    SCALE="${BALANCE_SCALE:-0.39269908169872414}"; STD_MIN="${MIN_STD:-0.03}"; STD_MAX="${MAX_STD:-0.15}"
    # Fixed start pose per robot: 1024 = 205 standing + 205/205/205/204 on poses 1-4 (eval: 8 = 2/2/1/2/1).
    TASK=(++env.mjx_task=standing env.mjx_simplified_rewards=false env.time_limit=5000
          "++env.pose_groups=${POSE_GROUPS:-[0,1,2,3,4]}"
          # Posture: pay for joints back at the standing pose (stand straight, not lean/awkward).
          "++env.mjx_standing_posture_weight=${POSTURE:-1.0}"
          'env.rsi_probabilities=[0.5,0.0,0.0,0.5]' 'env.eval_rsi_probabilities=[0.5,0.0,0.0,0.5]')
    # At +-22.5 deg the one-leg poses (3, 4, 7 need 70-75 deg) are clipped by the pose loader and
    # land in "other" without their lift, so balance trains standing + reachable pose variants;
    # one-leg support is learned in the walk stage (+-1 rad) as part of stepping.
    PROFILE=legacy ;;
  walk)
    LOGDIR="$ROOT/servo_gait_v3"; STEPS="${WALK_STEPS:-20000000}"
    INIT="${FROM_CKPT:-$ROOT/${BALANCE_RUN:-servo_balance}/latest.pt}"
    POSES=envs/mjx_walking/assets/manual_rsi_poses_merged.json
    # Residual walking: target = vendor gait reference + small correction, same +-22.5 deg range as
    # balance (the +-1 rad standing-relative walk collapsed twice: actions grew to ~11 deg error).
    SCALE="${WALK_SCALE:-0.39269908169872414}"
    # SCRIPTED=1 (default): follow scripts/scripted_walk.py's inverse-kinematics walk (it steps in this simulator; the
    # vendor pattern does not). It is quasi-static and slow, and tolerates only ~0.1-0.2 deg of action jitter, so
    # exploration is tiny, sway is not punished, and standing-still limits are long. SCRIPTED=0 = old vendor gait.
    if [ "${SCRIPTED:-1}" = 1 ]; then
      # HUMANLIKE=1 (default): the human-like walk (scripts/humanlike_walk.py: ZMP-planned gait + our 5-joint-leg IK), 6 cm/s,
      # 5 cm per foot per step, ~80 s open-loop in simulation. The environment itself adds the tilt stabiliser, so the
      # policy only learns small corrections (heading, speed, robustness). Training starts at a random phase of the walk
      # (85%); evaluation always starts standing. Joint speed cap 8 rad/s. HUMANLIKE=0 = the old slow quasi-static walk.
      if [ "${HUMANLIKE:-1}" = 1 ]; then
        SCRIPT_ARGS=(++env.mjx_scripted_gait=envs/mjx_walking/assets/humanlike_gait.npz ++env.mjx_lateral_scale="${LATERAL_SCALE:-0.08}"
                     ++env.mjx_scripted_phase_pool=envs/mjx_walking/assets/humanlike_phase_pool.npz
                     "++env.mjx_stabilizer=${STABILIZER:-[1.0,0.04,-1.0,0.25]}" ++env.mjx_phase_obs=true
                     ++env.mjx_control_slew_rad_s="${SLEW:-8.0}")
      else
        SCRIPT_ARGS=(++env.mjx_scripted_gait=envs/mjx_walking/assets/scripted_gait.npz ++env.mjx_lateral_scale="${LATERAL_SCALE:-0.15}"
                     ++env.mjx_scripted_phase_pool=envs/mjx_walking/assets/scripted_phase_pool.npz
                     ++env.mjx_phase_control="${PHASE_CONTROL:-0.5}" ++env.mjx_control_slew_rad_s="${SLEW:-8.0}")
      fi
      GAIT_SPEED_DEF=1.0; STALL_GRACE_DEF=8; STALL_OUT_DEF=20; RSI_DEF='[0.15,0.0,0.0,0.85]'; EVAL_RSI_DEF='[1.0,0.0,0.0,0.0]'; STD_MIN="${MIN_STD:-0.005}"; STD_MAX="${MAX_STD:-0.03}"; ACTION_COST_DEF=10
      [ -z "${WALK_SCALE:-}" ] && SCALE=0.1
    else
      SCRIPT_ARGS=(); GAIT_SPEED_DEF=0.5; STALL_GRACE_DEF=2.5; STALL_OUT_DEF=8; RSI_DEF='[0.6,0.0,0.0,0.4]'; EVAL_RSI_DEF='[0.6,0.0,0.0,0.4]'; STD_MIN="${MIN_STD:-0.01}"; STD_MAX="${MAX_STD:-0.06}"; ACTION_COST_DEF=2.0
    fi
    [ "$INIT" != null ] && EXTRA=(init_from_action_scale_compensation=true)
    if [ ! -f "$LOGDIR/latest.pt" ] && [ "$INIT" != null ] && [ ! -f "$INIT" ]; then
      echo "Walk stage needs the balance policy: $INIT not found (run STAGE=balance first, or FROM_CKPT=null)"; exit 2
    fi
    TASK=(++env.mjx_task=walking env.mjx_simplified_rewards=true env.time_limit="${TIME_LIMIT:-1000000000}" ++env.mjx_gait_residual=true ++env.mjx_alive_bonus=0.5
          ++env.mjx_fall_penalty="${FALL_PENALTY:-150}" ++env.mjx_landing_weight=10 ++env.mjx_first_step_bonus=20 ++env.mjx_goal_bonus=500 ++env.mjx_stall_out_penalty="${STALL_OUT_PENALTY:-30}"
          env.gait_reference_speed="${GAIT_SPEED:-$GAIT_SPEED_DEF}" ++env.mjx_stall_out_seconds="${STALL_OUT:-$STALL_OUT_DEF}" ++env.mjx_stall_grace_seconds="${STALL_GRACE:-$STALL_GRACE_DEF}" "${SCRIPT_ARGS[@]}" ++env.mjx_still_pays="${STILL_PAYS:-0}" ++env.mjx_action_weight="${ACTION_COST:-$ACTION_COST_DEF}"
          ++env.mjx_injected_pose_min_id="${INJECTED_MIN_ID:-8}" ++env.mjx_footsteps="${DOTS:-true}" ++env.mjx_dot_first_from_toe="${DOT_FIRST_FROM_TOE:-0.05}" ++env.mjx_dot_step="${DOT_STEP:-0.05}" ++env.mjx_dot_radius="${DOT_RADIUS:-0.025}" ++env.mjx_dot_first_side="${DOT_FIRST_SIDE:--1}" ++env.mjx_dot_first_behind_toe="${DOT_FIRST_BEHIND_TOE:-null}" ++env.mjx_dot_toe_target="${DOT_TOE:-1}" ++env.mjx_dot_max_dots="${DOT_MAX:-null}" ++env.mjx_dot_end_on_success="${DOT_END:-0}"
          "env.rsi_probabilities=${WALK_RSI:-$RSI_DEF}" "env.eval_rsi_probabilities=${EVAL_RSI:-$EVAL_RSI_DEF}")
    PROFILE=goal_corridor_v3 ;;
  *) echo "STAGE must be balance, walk or both"; exit 2 ;;
esac
RESUME=()
if [ -f "$LOGDIR/latest.pt" ]; then RESUME=("resume_from=$LOGDIR/latest.pt"); INIT=null; EXTRA=(); echo "Resuming $STAGE from $LOGDIR/latest.pt"; fi
echo "Stage $STAGE | run dir $LOGDIR | init $INIT ${EXTRA[*]} | worlds $ENV_NUM | steps $STEPS | train_ratio ${TRAIN_RATIO:-64} | buffer ${BUFFER:-524288} | action range ±$SCALE rad | std $STD_MIN-$STD_MAX | eval/ckpt every $EVERY | video $VIDEO ($GL)"
set +e
FROM_CKPT="$INIT" ENV_NUM="$ENV_NUM" STEPS="$STEPS" CKPT_EVERY="$EVERY" REWARD_PROFILE="$PROFILE" \
MIN_STD="$STD_MIN" MAX_STD="$STD_MAX" \
bash scripts/deploy/start_interactive_walking.sh "$LOGDIR" \
  env.pose_rsi_json="$POSES" "${TASK[@]}" \
  env.mjx_arm_action_scale="$SCALE" env.mjx_leg_action_scale="$SCALE" env.mjx_other_action_scale="$SCALE" \
  env.mjx_step_clearance_m="${STEP_CLEARANCE_M:-0.015}" \
  env.train_ratio="${TRAIN_RATIO:-64}" model.warmup=100 buffer.max_size="${BUFFER:-524288}" \
  trainer.learning_start_deadline_steps="$DEADLINE" \
  trainer.best_metric_key=score trainer.checkpoint_order=task_first \
  trainer.eval_every="$EVERY" trainer.eval_max_steps=3000 \
  trainer.eval_video="$([ "$VIDEO" = 1 ] && echo true || echo false)" ++trainer.eval_video_every="$EVERY" \
  ++trainer.eval_video_save_mp4=true ++trainer.eval_video_tensorboard=false ++trainer.eval_video_stride=10 \
  ++trainer.eval_video_width=320 ++trainer.eval_video_height=240 \
  "${EXTRA[@]}" "${RESUME[@]}"
code=$?
set -e
echo "Stage $STAGE finished with exit code $code"
exit $code
