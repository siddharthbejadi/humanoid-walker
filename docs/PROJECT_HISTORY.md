# Project history: what was tried, what failed, what is left

Distilled from the project notes kept during development (4 to 14 Oct 2026). Hosts, credentials and machine-specific paths are deliberately omitted.
Companion documents: [`PROJECT_NOTES_12Oct.md`](PROJECT_NOTES_12Oct.md), [`../deploy/DEPLOYMENT_SPEC.md`](../deploy/DEPLOYMENT_SPEC.md), [`../r2dreamer/docs/`](../r2dreamer/docs).

## Pipeline status (plain language)

```
[DONE]  simulator + reward + tests
[DONE]  scripted human-like gait (expert)            -> works only at 18 kg·cm torque
[DONE]  collect 64 expert episodes, 40 ms delay, IMU noise, masked sensors
[DONE]  behaviour-clone student (12k updates)       -> 97-100 % of simulated robots walk, 0-3 % falls
[NOW ]  robot bring-up: servo calibration, IMU axes, 50 Hz command path, relax()
[NEXT]  Gates 0-4 of deploy/ROBOT_TEST_PROTOCOL.md  -> physical walk
[LATER] retrain with measured torque (1.08 N·m knees) and per-joint servo models
```
Reinforcement learning is **not** used in the final policy (see dead ends).

## Timeline

| Date | Event |
|---|---|
| 4 to 7 Oct | R2Dreamer learner with an MJX-Warp teacher environment; standing curricula, capacity probes, interactive pose-injection editor (`scripts/interactive_rsi.py`). |
| 8 Oct | Limp-servo root cause fixed in the scene (servo kp); goal-corridor reward v3 with 85 reward sanity tests. Long RL runs (up to 14 M steps) still never produced a valid step. |
| 10 Oct | Hardware audit: torso IMU ICM-20948, head IMU ISM330DLC, firmware streams raw counts at a nominal 50 Hz with no fusion. |
| 10 to 12 Oct | The vendor gait cannot step in simulation. Wrote a scripted IK walk (slow) and the human-like ZMP walk (6.2 cm/s). User's hardware bundle measured: IMU axes, MG996R knees at about 1.08 N·m, 40 to 50 Hz IMU rate, no servo feedback. |
| 13 Oct | Script-free student by behaviour cloning. Step-counter bug found and fixed. Realistic variant (masked sensors, 40 ms delay, IMU noise, 6 rad/s slew) trained and evaluated. Deployment spec and robot host written. |
| 14 Oct | Stand-still IMU calibration added to the host; first real at-rest capture showed a 14 to 16 degree tilt versus the simulator standing pose; servo calibration notes collected; this repo assembled. |

## Key findings

* **Only behaviour cloning works.** Any RL actor update, even a tiny one, or an encoder/RSSM update on a fresh replay broke the cloned gait in fewer than 70 updates. The actor standard deviation is fixed by the contract to 0.0003 to 0.002, so the REINFORCE gradient scales like 1/sigma^2. Freezing encoder/RSSM with the policy loss off held, but gave no improvement.
* **DART (noisy-action demonstrations) made students worse** (0 to 75 % walking).
* **"0 valid steps" was a counter bug, not a gait failure:** `foot_landing_events` in `envs/mjx_walking/env.py` required the opposite foot's load to exceed 0.55 at touchdown, but the scripted walk shifts weight early (0.34 to 0.42). The threshold `LANDING_SUPPORT_LOAD` is now 0.30 (the 15 mm clearance threshold is unchanged). The scripted walk then scores about 21 left / 22 right valid steps.
* **Closed-loop results (3 seeds x 32 worlds, 20 s, standing start):**
  original gait, slew 8: 100 % walking, 0 falls (5k updates);
  tuned gait at slew 6: 5k updates 82 % walking with 15 % falls, 12k updates 78 to 88 % with 0 falls, about 1 in 5 worlds never starts;
  realistic student (final, `student_realistic_12k.pt`): about 100 / 100 / 97 % walking, falls 0 / 0 / 3 %, about 20 valid steps per foot.
  "Walking" here is the project's strict detector, not proof of a good walk. `walking_qualified` stays false.
* **Over-sampling episode-start windows (`STARTUP_FRACTION=0.5`) made students stand still** (0 % walking). Keep the default 0.25.
* **Joint range:** the gait needs hip angles up to about 0.95 rad (54 degrees); the firmware default servo bounds (about +-36 degrees) are too narrow.

## Dead ends (do not retry without changing the premise)

1. RL fine-tuning of the student (breaks the cloned gait; needs a different actor-noise contract).
2. DART noise injection for the demonstrations.
3. Over-weighting episode starts in behaviour cloning.
4. Training on the vendor gait (cannot step in this simulator).
5. Relying on the 12 Oct scripted walk under measured torque: it falls (see `figures/fig_stress.png`).

## Working agreements

* The simulator deliberately uses **18 kg·cm (1.77 N·m) servos on all joints**, a user decision, even though the knees measure about 1.08 N·m. Revisit before expecting the physical robot to match.
* Sensors the robot does not have are masked in the observation: foot pressure (8 inputs) and the magnetometer (3 inputs).
* Progress updates should include a short DONE / NOW / NEXT flowchart through to the physical robot and a plain statement of whether RL is in use.
* Work on a Linux filesystem copy of the project, not on the Windows mount (slow).

## Hardware facts and open items

* Board: STM32N6570-DK; one ST-LINK virtual COM port for IMU in and servo frames out; two PCA9685 drivers at 50 Hz (about 4.88 us per count).
* Torso IMU frame measured as FLU = (-rawX, -rawY, +rawZ); simulator native = (-rawY, +rawX, +rawZ). At rest |a| = 1.010 g, confirming the +-2 g scale.
* Arms: channel 2 is the arm, channel 1 the elbow. The simulator's hand-tip joint versus the firmware "gripper" channel is **unconfirmed**.
* Default servo centres (pulse widths) recorded by the user: thighs and knees 1600, right hip roll 2100, left hip roll 1700 (not lower: collision), feet right 1600 / left 1700, shoulders right 1000 / left 2000.
* Open: foot direction, left-knee note, right lower-thigh neutral (1500 vs 1600), head and hand neutrals, microseconds per radian, full-loop latency, loaded supply voltage and current, true mass (about 2.66 kg vs 3.0 kg modelled).
