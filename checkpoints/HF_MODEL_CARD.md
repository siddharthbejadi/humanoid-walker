---
library_name: pytorch
tags:
  - robotics
  - humanoid
  - behavior-cloning
  - mujoco
  - mjx
  - dreamer
pipeline_tag: robotics
---

# humanoid-walker-student

Recurrent (R2Dreamer / DreamerV3-style) walking policy for a 19-servo humanoid, trained by **behaviour cloning** of a scripted human-like gait in MuJoCo / MJX-Warp.
Input: 45 proprioceptive numbers at 50 Hz (last command + IMU, magnetometer and foot sensors masked). Output: 19 servo targets in [-1, 1].
Code, training recipe and deployment spec: https://github.com/siddharthbejadi/humanoid-walker (source commit `2e23f21` or later).

> **Simulation only.** Not tested on the physical robot. The evaluation harness reports `walking_qualified: false`.
> The simulator assumes 1.77 N·m on every joint; the real knees measure about 1.08 N·m, and the scripted expert falls under measured limits.

## Files

| File | Size | SHA-256 |
|---|---|---|
| `student_realistic_12k.pt` | 112,072,779 bytes | `1d1af540214b7be6f201eed97bd47c0adca665b309b230968688cf1a5db806ce` |

## Results (simulation, standing start, 20 s, 32 robots per seed)

| Eval seed | Forward distance | Valid steps | Falls |
|---|---|---|---|
| 1100 | 1.40 m | 41 | 0 % |
| 1200 | 1.36 m | 40 | 0 % |
| 1300 | 1.32 m | 39 | 3 % |

Held-out behaviour-cloning error: 2.9e-6 action MSE after 12 000 updates (1.7e-2 at initialisation). The walk is slow (about 7 cm/s).
"Realistic" means realistic sensing and latency (40 ms command delay, IMU noise, 6 rad/s slew, masked sensors), not realistic torque.

## Use

```bash
git clone https://github.com/siddharthbejadi/humanoid-walker && cd humanoid-walker
hf download siddharthbejadi/humanoid-walker-student student_realistic_12k.pt --local-dir checkpoints
sha256sum checkpoints/student_realistic_12k.pt        # must match the table above
# GPU machine, after r2dreamer/scripts/deploy/setup_script_free.sh:
cd r2dreamer && .venv/bin/python ../deploy/watch_policy.py --repo . \
    --checkpoint ../checkpoints/student_realistic_12k.pt --seconds 20 --seed 1100 --video ../walk.mp4
```

The checkpoint binds a training contract (`deploy/contract_clean12k.json` in the GitHub repo) and refuses to load against a simulator whose contract differs.

## Limitations

Behaviour-cloned copy of a scripted gait, not an independent learner (RL fine-tuning breaks it); servo calibration and IMU axes unverified on hardware;
mass, supply sag, backlash and friction not modelled. License: pending confirmation of third-party components (see `THIRD_PARTY_NOTICES.md` in the GitHub repo).
