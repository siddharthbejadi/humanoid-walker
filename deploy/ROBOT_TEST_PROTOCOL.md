# Staged robot test protocol (do in order; stop at the first failed gate)

Nothing below has been performed. Every stage needs a person at the power switch.

**Gate 0 — bench, no policy**
1. Measure per-joint neutral/sign/µs-per-rad; command ±10° on each joint; confirm direction matches the simulator
   (positive angle = same direction as in the model). Record in a table.
2. Measure real command→motion latency (command a step, record with IMU or video). Pass: ≤ 40 ms. If larger, retrain.
3. Implement `ServoLink.relax()` (cut torque). Test it. Pass: robot goes limp within 100 ms of the call.
4. IMU: stand level and still 10 s → accel ≈ (0, 0, 9.8) in the sim frame; lean forward 20° → check sign/axis;
   record gyro bias. Confirm the sample age stays < 60 ms.

**Gate 1 — plumbing, servos disabled** (`robot_host.py --dry-run`)
Run 60 s on the real IMU with the robot held still: no safe-stops, loop overruns = 0, commands stay finite and
within ±0.2 rad of standing (a still robot should not demand large moves).

**Gate 2 — hanging, feet off the ground, torque limited**
Robot suspended in a harness, feet ~5 cm above the floor. Servos enabled, power supply current-limited.
Run 10 s. Expect: leg swing resembling a walking cycle. Pass: no jerking, no servo stall/over-current, `relax()` works.

**Gate 3 — harness on the ground (light support, slack rope)**
Feet on the floor, harness taking weight only if it tips. Run 10 s, then 20 s. Pass: stays upright, steps with both feet.
Abort immediately on tilt > 20°, any servo overheating, or a stuck joint.

**Gate 4 — free walk on a flat, high-friction surface**, 1 m marked course, soft landing, spotter, 3 trials.

If a gate fails, do not skip ahead: log the IMU/command CSV and bring it back — the simulator can be re-fitted
(torque, latency, noise) from those logs.
