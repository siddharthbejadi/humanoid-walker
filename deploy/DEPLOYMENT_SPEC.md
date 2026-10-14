# Script-free walking student — deployment spec (simulator contract → robot)

**Status (13 Oct 2026):** simulation only. Nothing here has been run on the physical robot. `robot_host.py`
has been exercised with a synthetic IMU and dry-run servos only.

## 1. What the policy is
A recurrent network (R2Dreamer world-model backbone, actor head only used at run time) trained by
**behaviour cloning** of the scripted human-like gait. **No RL is used.** It runs at **50 Hz** (20 ms).
Input: 45 numbers. Output: 19 numbers in [-1, 1].

## 2. Output → servos
`target_rad = (π/2) · action`, for each of the 19 joints (centre 0, scale π/2, range ±π/2).
Then clip to the joint limits, then apply the **6 rad/s slew limit** (max 0.12 rad change per 20 ms).
Joint order (index: name): 0 L hip-torso(roll), 1 L hip-upper(pitch), 2 L lower-thigh bracket, 3 L knee, 4 L foot,
5–9 the same for R, 10 L shoulder, 11 L arm, 12 L elbow, 13–15 R shoulder/arm/elbow, 16 head, 17 L hand tip, 18 R hand tip.
Standing pose (sim zero ≈ standing): values in `contract_clean12k.json → control.standing_ctrl_rad` (all |x| < 0.05 rad).

**You must supply** (not known to me): per joint, servo neutral µs, sign, µs-per-rad and hard limits.
Known issue: the firmware default bounds (1200–1800 µs, ≈ ±36°) are narrower than the gait needs
(hip-pitch/thigh ≈ ±46–51°, knee ≈ 25°). Joint-direction conventions of the sim vs the robot are **unchecked**.

## 3. Input layout (45 channels, in order)
| Index | Channel | On the robot |
|---|---|---|
| 0–18 | last command, `ctrl/(π/2)` | host computes it (see §4) |
| 19–21 | torso accel (m/s², sim native frame) | ICM-20948, remapped (§5) |
| 22–24 | torso gyro (rad/s) | ICM-20948, bias-subtracted, remapped |
| 25–27 | magnetometer | **zeros** (masked in training) |
| 28–30 | "onboard" (head) accel | torso accel re-ordered to the head IMU axis order `(x, z, −y)` (head yaw assumed 0; head IMU not read) |
| 31–33 | "onboard" (head) gyro | torso gyro re-ordered `(x, z, −y)` |
| 34–36 | constant `[0.08, 0, 0]` | constant |
| 37–44 | foot pressure (8) | **zeros** (masked in training) |

## 4. Command latency
The student was trained with a **40 ms (2-step) command delay**. `last_command` in the simulator is the command
that was *applied* — i.e. sent 2 steps earlier. The host mirrors this by feeding the ctrl it sent 2 steps ago.
Do **not** add an artificial delay on top of the real latency; measure the real latency and compare with 40 ms.
If it is much larger (> 60 ms) retrain with that value.

## 5. IMU frame and units
Firmware prints raw counts (±2 g → 16384/g, ±250 dps → 131/dps). Measured torso frame (qualitative):
FLU = (−rawX, −rawY, +rawZ). Simulator native frame is (+X left, +Y backward, +Z up), so
`sim = (−rawY, +rawX, +rawZ)` for both accel and gyro. Gyro bias: average 2 s while standing still.
The real IMU log rate was 40–50 Hz, not exactly 50. The axis result is qualitative and the standing-level
calibration was never done: **verify with a lean test** (lean forward → sim-frame accel Y should go negative…
confirm sign by eye before trusting it).

## 6. Not modelled in training (risk list)
- servo torque: sim assumes 18 kg·cm (1.77 N·m) on all joints (your instruction); your knees measure ≈ 1.08 N·m.
  With 1.08 N·m on all joints the scripted gait fell in simulation (and a search found no walking gait).
- sensor feedback: the robot has no joint-position or current feedback; the sim student also has none.
- supply sag, foot friction, real mass (≈ 2.66 kg vs 3.0 kg modelled), backlash.
- head IMU, head-yaw dependence of the "onboard" slots.
- clock drift between IMU samples and the control loop.

## 7. Files
`robot_host.py` (runner), `ROBOT_TEST_PROTOCOL.md` (staged tests), `contract_clean12k.json` (exact contract numbers),
checkpoint (see report), source tree of the training code (needed to load the checkpoint).

## 8. Stand-still calibration (added 14 Oct 2026, after the first real IMU capture)
Real capture at rest (robot state unknown): raw ≈ (−4556, −529, 15906) counts → |a| = 1.010 g (confirms ±2 g scale),
which in the simulator frame is (0.32, −2.73, 9.52) m/s² — a **14–16° tilt** from the simulator's standing reading
`(−0.108, −0.303, 9.802)` m/s² (measured by holding the standing command in the simulator, noise off; gyro 0).
`robot_host.py` therefore averages the IMU for 2 s at start, computes the rotation that maps the measured gravity
direction onto the simulator's standing direction, and applies it to accel and gyro. It aborts if |a| is not within 1.5
m/s² of 9.81 or if the correction exceeds 25°. **Only valid if the robot is upright in the neutral standing pose
during those 2 s** (check with a level or a phone). If the real torso is actually leaning, the calibration would hide it.
