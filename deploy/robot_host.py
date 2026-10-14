#!/usr/bin/env python3
"""Host-side 50 Hz runner for the script-free walking student.

STATUS: written against the simulator contract, self-tested with a synthetic IMU only.
It has NOT been run on the physical robot. Read DEPLOYMENT_SPEC.md before connecting servos.

Loop (every 20 ms): read torso IMU -> build the 45-value observation exactly as the
simulator did -> student.act() -> joint targets -> 6 rad/s slew limit -> clip -> send.

Run a plumbing check without hardware:
    python robot_host.py --repo /path/to/r2dreamer --checkpoint latest.pt --selftest
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---- constants from the training contract (verified against the checkpoint at start-up) ----------------
N_JOINTS = 19
SCALE = math.pi / 2          # action 1.0 == +90 deg; center 0 for every joint
CONTROL_DT = 0.02            # 50 Hz
SLEW_RAD_S = 6.0             # real servo speed 5.5-7 rad/s
DELAY_STEPS = 2              # latency the student was trained with (40 ms)
TARGET_FORWARD_SPEED = 0.08  # constant observation channel
G = 9.80665

# Raw ICM-20948 counts -> SI (firmware: +-2 g, +-250 dps, raw int16, no fusion)
ACCEL_COUNTS_PER_G = 16384.0
GYRO_COUNTS_PER_DPS = 131.0

# Measured torso IMU frame: FLU = (-rawX, -rawY, +rawZ).  The simulator feeds its NATIVE sensor frame
# (+X left, +Y backward, +Z up) = (FLU_y, -FLU_x, FLU_z) = (-rawY, +rawX, +rawZ).
def raw_to_sim_frame(v):
    return np.array([-v[1], v[0], v[2]], dtype=np.float32)


class Imu:
    """Subclass and return (accel_raw[3], gyro_raw[3], age_seconds) from the newest valid sample."""

    def latest(self):
        raise NotImplementedError


class SyntheticImu(Imu):
    """Standing still: 1 g up, zero rotation. Plumbing test only."""

    def latest(self):
        return np.array([0., 0., ACCEL_COUNTS_PER_G]), np.zeros(3), 0.0


class SerialImu(Imu):
    """Parses the existing firmware line:  IMU,imu_ms,read_ok,ax,ay,az,gx,gy,gz,mx,my,mz,mag_valid"""

    def __init__(self, port, baud=115200):
        """`port` is a COM-port name OR an already-open pyserial object (shared with the servo link)."""
        if isinstance(port, str):
            import serial  # pyserial
            port = serial.Serial(port, baud, timeout=0.001, write_timeout=0.02)
        self.link = port
        self.buffer = b""
        self.sample = None
        self.stamp = 0.0
        self.state, self.state_stamp = None, 0.0     # last "WALKER,STATE,<armed|run>" heartbeat from the STM32

    def latest(self):
        self.buffer += self.link.read(4096)
        *lines, self.buffer = self.buffer.split(b"\n")
        for line in lines:
            parts = line.strip().decode("ascii", "ignore").split(",")
            if len(parts) >= 3 and parts[0].endswith("WALKER") and parts[1] == "STATE" or (len(parts) >= 3 and parts[0] == "WALKER" and parts[1] == "STATE"):
                self.state, self.state_stamp = parts[2].strip(), time.monotonic()
                continue
            if len(parts) >= 13 and parts[0] == "IMU" and parts[2] == "1":
                try:
                    values = [float(x) for x in parts[3:9]]
                except ValueError:
                    continue
                self.sample, self.stamp = (np.array(values[:3]), np.array(values[3:6])), time.monotonic()
        if self.sample is None:
            return None, None, float("inf")
        return self.sample[0], self.sample[1], time.monotonic() - self.stamp


class ServoLink:
    """Subclass for the real servo path. send() takes 19 joint angles in radians (sim zero = standing)."""

    def send(self, angles_rad):
        raise NotImplementedError

    def relax(self):
        """Cut torque / release servos (explicit request only)."""
        raise NotImplementedError

    def hold(self):
        """Stop sending: the STM32 watchdog then HOLDS every servo at its last pulse (stays stiff)."""


class DryRunServoLink(ServoLink):
    def __init__(self):
        self.last = None
        self.sent = 0

    def send(self, angles_rad):
        self.last, self.sent = np.array(angles_rad), self.sent + 1

    def relax(self):
        print("[dry-run] relax() called")

    def hold(self):
        print("[dry-run] hold() called")


# Simulator accelerometer at standing, measured by holding the standing command for 1 s (noise off):
#   torso (-0.108, -0.303, 9.802) m/s^2, gyro 0.  The calibration below makes the real IMU read the same direction.
SIM_STAND_ACCEL = np.array([-0.108, -0.303, 9.802], dtype=np.float64)


def rotation_between(a, b):
    """Rotation matrix that turns direction a into direction b (Rodrigues)."""
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    axis, cos = np.cross(a, b), float(np.dot(a, b))
    sin = float(np.linalg.norm(axis))
    if sin < 1e-9:
        return np.eye(3)
    axis /= sin
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + sin * k + (1. - cos) * (k @ k)


def head_frame(v):
    """The simulator's head IMU frame is (+X left, +Y up, +Z forward); torso is (+X left, +Y backward, +Z up)."""
    return np.array([v[0], v[2], -v[1]], dtype=np.float32)


class PreviewServoLink(ServoLink):
    """Watch-only: converts the policy's joint angles into servo pulses with the calibration table, sends nothing,
    and reports the range of pulses per joint (and any that would hit the limits) at the end."""

    def __init__(self, calibration_json):
        self.table = json.loads(Path(calibration_json).read_text())
        self.low = np.full(N_JOINTS, np.inf)
        self.high = np.full(N_JOINTS, -np.inf)

    def send(self, angles_rad):
        t = self.table
        pulse = np.array([e["neutral_us"] + e["sign"] * e["us_per_rad"] * a for e, a in zip(t, angles_rad)])
        self.low, self.high = np.minimum(self.low, pulse), np.maximum(self.high, pulse)

    def relax(self):
        print("[preview] relax() called")

    def hold(self):
        print("[preview] hold() called")

    def report(self):
        print("\nPulses the policy would command (watch-only, nothing was sent):")
        print(f"{'joint':14s} {'neutral':>7s} {'would use':>15s} {'limits':>11s}")
        for e, lo, hi in zip(self.table, self.low, self.high):
            if e.get("unused"):
                print(f"{e['sim_joint']:14s} {'-':>7s} {'(not driven: left limp)':>15s}")
                continue
            clipped = "  <-- CLIPPED" if lo < e["min_us"] - 3 or hi > e["max_us"] + 3 else ""   # 3 us tolerance: held joints sit on a limit
            print(f"{e['sim_joint']:14s} {e['neutral_us']:7d} {lo:7.0f}..{hi:<7.0f} {e['min_us']:5d}..{e['max_us']:<5d}{clipped}")


def build_observation(last_command, accel_sim, gyro_sim):
    """45 channels in simulator order. FSR (8) and magnetometer (3) are zero: masked in training."""
    obs = np.zeros(45, dtype=np.float32)
    obs[0:19] = last_command
    obs[19:22], obs[22:25] = accel_sim, gyro_sim
    # obs[25:28] magnetometer: masked (zeros)
    obs[28:31], obs[31:34] = head_frame(accel_sim), head_frame(gyro_sim)   # head-IMU slots: same motion, head axis order
    obs[34] = TARGET_FORWARD_SPEED
    # obs[37:45] foot pressure: masked (zeros)
    return obs


def tilt_degrees(accel_sim):
    norm = float(np.linalg.norm(accel_sim))
    return 180.0 if norm < 1e-6 else math.degrees(math.acos(max(-1., min(1., accel_sim[2] / norm))))


def run(args):
    def fault(message, code):
        print(message)
        if args.on_fault == "release":
            servo.relax(); print("servos RELEASED (limp). Console: 'off all'.")
        else:
            servo.hold(); print("servos HOLD their last pulses (stiff). Console: 'off all' releases, 'stand' goes to neutral.")
        return code

    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch
    from script_free_policy import ScriptFreePolicy

    if sys.platform == "win32":                       # default Windows timer tick is ~15.6 ms: ask for 1 ms
        try:
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass
    torch.set_num_threads(args.threads)
    policy = ScriptFreePolicy(args.checkpoint, device=args.device)
    contract = policy.contract
    ctrl = contract["control"]
    low = np.asarray(ctrl["actuator_ctrlrange_rad"], np.float32)[:, 0]
    high = np.asarray(ctrl["actuator_ctrlrange_rad"], np.float32)[:, 1]
    stand = np.asarray(ctrl["standing_ctrl_rad"], np.float32)
    assert abs(ctrl["control_dt_seconds"] - CONTROL_DT) < 1e-9, "checkpoint control period is not 20 ms"
    assert np.allclose(ctrl["command_center_rad"], 0.) and np.allclose(ctrl["action_scale_rad"], SCALE, atol=1e-4)
    assert abs(ctrl["slew_rad_per_second"] - SLEW_RAD_S) < 1e-6, "checkpoint trained with a different slew limit"
    realism = contract.get("realism", {})
    if realism.get("command_delay_steps", 0) != DELAY_STEPS:
        print(f"WARNING: checkpoint was trained with command delay {realism.get('command_delay_steps', 0)} steps, "
              f"host assumes {DELAY_STEPS}")
    mask = contract.get("observation_mask", {})
    if not (mask.get("fsr") and mask.get("mag")):
        print("WARNING: checkpoint was NOT trained with foot-pressure/magnetometer masked; the real robot cannot "
              "provide those inputs.")

    if args.selftest:
        imu = SyntheticImu()
        servo = PreviewServoLink(args.calibration) if args.calibration else DryRunServoLink()
    else:
        if not args.port:
            raise SystemExit("--port COMx (the ST-LINK virtual COM port) is required unless --selftest")
        import serial  # pyserial
        link = serial.Serial(args.port, args.baud, timeout=0.001, write_timeout=0.02)  # ONE port: IMU in, servo frames out
        imu = SerialImu(link)
        link.write(b"\r\nimu on\r\n")   # the STM32 starts with the IMU rows off (harmless if already armed)
        if args.dry_run:
            servo = PreviewServoLink(args.calibration) if args.calibration else DryRunServoLink()
        else:
            if not args.calibration:
                raise SystemExit("--calibration servo_calibration_sim.json is required to drive servos")
            from serial_servo_link import SerialServoLink
            servo = SerialServoLink(link, args.calibration)
            print("LIVE servo mode. Before starting: servo supply ON, robot supported, console `stand` done, then `run confirm` "
                  "(WALKER,ARMED), then close the terminal. Starting in 3 s - be ready at the power switch.")
            time.sleep(3.)

    live = not (args.selftest or args.dry_run)
    if not 1.0 <= args.slow <= 3.0:
        raise SystemExit("--slow must be between 1 and 3 (the STM32 watchdog releases the servos after 100 ms without a frame)")
    period = CONTROL_DT * args.slow
    if live:
        t_end = time.monotonic() + 3.0
        while time.monotonic() < t_end and getattr(imu, "state", None) not in ("armed", "run"):
            imu.latest()
            time.sleep(0.02)
        if getattr(imu, "state", None) not in ("armed", "run"):
            print("STOP: the STM32 is not armed (no WALKER,STATE heartbeat in 3 s). Nothing was sent.\n"
                  "  Console: `stand`, then `run confirm` (it must print WALKER,ARMED), CLOSE the terminal, then start this script.\n"
                  "  After any stop or relax you must repeat `stand` and `run confirm`.")
            return 9

    # Stand-still calibration (2 s). The robot must be stationary, UPRIGHT and in the neutral (standing) pose.
    bias, accel_sum, count, t_end = np.zeros(3), np.zeros(3), 0, time.monotonic() + (0. if args.selftest else 2.)
    while True:
        accel_raw, gyro, age = imu.latest()
        if gyro is not None and age < 0.1:
            bias, accel_sum, count = bias + gyro, accel_sum + accel_raw, count + 1
        if time.monotonic() >= t_end:
            break
        time.sleep(CONTROL_DT)
    if count < (1 if args.selftest else 25):
        return fault("SAFE STOP: no IMU data during stand-still calibration", 6)
    bias, accel_mean = bias / count, accel_sum / count
    sim_mean = raw_to_sim_frame(accel_mean / ACCEL_COUNTS_PER_G * G).astype(np.float64)
    mount = rotation_between(sim_mean, SIM_STAND_ACCEL).astype(np.float32)   # removes the IMU mounting tilt
    if args.freeze_imu:
        print("FREEZE-IMU: the policy is fed a still, upright IMU; the real IMU only checks that the board is alive.")
    mount_deg = math.degrees(math.acos(max(-1., min(1., (np.trace(mount) - 1.) / 2.))))
    print(f"stand-still calibration: |a| = {np.linalg.norm(sim_mean):.2f} m/s^2, mounting/lean correction {mount_deg:.1f} deg")
    if abs(np.linalg.norm(sim_mean) - G) > 1.5:
        return fault("SAFE STOP: gravity magnitude is wrong - check IMU units/scale", 7)
    if mount_deg > 25.:
        return fault("SAFE STOP: the robot is leaning more than 25 deg during calibration - stand it upright", 8)

    sent = collections.deque([stand.copy()] * (DELAY_STEPS + 1), maxlen=DELAY_STEPS + 1)  # sent ctrl history
    ctrl_now = stand.copy()
    # Warm up the network and the timer: the first calls are much slower than the steady state.
    for _ in range(30):
        policy.act(np.zeros((1, 45), np.float32), torch.tensor([[True]]))
    policy.reset(1)
    is_first = True
    log, overruns, steps = [], 0, int(args.seconds / CONTROL_DT)
    t_imu, t_inf, t_send, t_tot = [], [], [], []
    window = collections.deque(maxlen=25)            # recent late/not-late flags
    next_tick = time.perf_counter()
    for k in range(steps):
        loop_start = time.perf_counter()
        accel_raw, gyro_raw, age = imu.latest()
        t_imu.append(1000. * (time.perf_counter() - loop_start))
        if accel_raw is None or age > 0.06:
            return fault("SAFE STOP: IMU stale", 2)
        if live and time.monotonic() - getattr(imu, "state_stamp", time.monotonic()) > 2.5:
            return fault("SAFE STOP: the STM32 stopped reporting armed/run (it released the servos?)", 10)
        if args.freeze_imu:
            accel, gyro = SIM_STAND_ACCEL.astype(np.float32), np.zeros(3, np.float32)
        else:
            accel = mount @ raw_to_sim_frame(accel_raw / ACCEL_COUNTS_PER_G * G)
            gyro = mount @ raw_to_sim_frame((gyro_raw - bias) / GYRO_COUNTS_PER_DPS * math.pi / 180.)
            if tilt_degrees(accel) > args.max_tilt_deg:
                return fault(f"SAFE STOP: tilt {tilt_degrees(accel):.1f} deg", 3)
        # Simulator semantics: last_command is the ctrl that was APPLIED, i.e. the one sent DELAY_STEPS ago.
        last_command = np.clip(sent[0] / SCALE, -1., 1.)
        obs = build_observation(last_command, accel, gyro)
        t0 = time.perf_counter()
        action = policy.act(obs[None], torch.tensor([[is_first]]))[0].cpu().numpy()
        t_inf.append(1000. * (time.perf_counter() - t0))
        is_first = False
        if not np.isfinite(action).all():
            return fault("SAFE STOP: non-finite action", 4)
        target = np.clip(SCALE * action, low, high)
        step = SLEW_RAD_S * CONTROL_DT
        ctrl_now = np.clip(target, ctrl_now - step, ctrl_now + step)
        sent.append(ctrl_now.copy())
        t0 = time.perf_counter()
        servo.send(ctrl_now)
        t_send.append(1000. * (time.perf_counter() - t0))
        log.append(np.concatenate([[k], ctrl_now]))
        next_tick += period
        now = time.perf_counter()
        if not args.selftest:
            late = now > next_tick
            window.append(late)
            if late:
                overruns += 1
                next_tick = now                       # resynchronise instead of racing to catch up
                if live and sum(window) >= 5:         # sustained lateness only matters when servos are being driven
                    return fault("SAFE STOP: loop overrun (5 late steps in the last 25)", 5)
            else:                                     # sleep most of the slack, spin the last ~2 ms for accuracy
                while True:
                    remaining = next_tick - time.perf_counter()
                    if remaining <= 0:
                        break
                    if remaining > 0.003:
                        time.sleep(remaining - 0.002)
        t_tot.append(1000. * (time.perf_counter() - loop_start))
    arr = np.array(log)
    stat = lambda v: dict(mean=round(float(np.mean(v)), 2), p99=round(float(np.percentile(v, 99)), 2), max=round(float(np.max(v)), 2))
    print(json.dumps(dict(steps=len(log), late_steps=overruns, over_period_plus_1ms=int(sum(1 for v in t_tot if v > 1000. * period + 1.)),
                          step_ms=stat(t_tot), imu_read_ms=stat(t_imu), policy_ms=stat(t_inf), send_ms=stat(t_send),
                          finite=bool(np.isfinite(arr).all()), max_abs_angle_rad=round(float(np.abs(arr[:, 1:]).max()), 3),
                          max_step_rad=round(float(np.abs(np.diff(arr[:, 1:], axis=0)).max()), 3))))
    if hasattr(servo, "report"):
        servo.report()
    if args.log:
        np.save(args.log, arr)
    if live:
        # Soft stop: ramp every joint back to neutral at the slew limit (about 1 s), THEN release the servos,
        # instead of letting the STM32 watchdog drop the robot limp mid-motion.
        print("soft stop: returning to neutral...")
        step = SLEW_RAD_S * CONTROL_DT
        for _ in range(int(2. / CONTROL_DT)):
            ctrl_now = np.clip(stand, ctrl_now - step, ctrl_now + step)
            servo.send(ctrl_now)
            time.sleep(period)
            if np.allclose(ctrl_now, stand, atol=1e-3):
                break
        time.sleep(0.3)
        if args.release_at_end:
            servo.relax()
            print("released (servos limp). Console: type 'off all'; switch the servo supply off.")
        else:
            servo.hold()
            print("holding neutral (servos stay stiff; the STM32 keeps the last pulses). Console: 'off all' releases them.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="r2dreamer source tree (contains script_free_policy.py)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4), help="torch CPU threads (measured on the robot laptop: 1 thread 15 ms, 4 threads 8 ms, 8 threads 7 ms per step)")
    ap.add_argument("--port", help="ST-LINK virtual COM port, e.g. COM5 (IMU in, servo frames out, one port)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--calibration", help="servo_calibration_sim.json from tools/csv_to_limits.py")
    ap.add_argument("--seconds", type=float, default=20.)
    ap.add_argument("--max-tilt-deg", type=float, default=35.)
    ap.add_argument("--slow", type=float, default=1.0, help="run the gait this many times slower (1..3): control period 20 ms x N, joint speeds / N")
    ap.add_argument("--on-fault", choices=("hold", "release"), default="hold",
                    help="on a SAFE STOP: hold (default; servos stay stiff at their last pulses) or release (limp)")
    ap.add_argument("--release-at-end", action="store_true", help="after the soft stop, release the servos (default: keep holding neutral)")
    ap.add_argument("--freeze-imu", action="store_true", help="feed the policy a still, upright IMU (no feedback from the real robot): "
                    "use while HANGING, where the swinging body would otherwise feed back into the gait")
    ap.add_argument("--selftest", action="store_true", help="synthetic IMU, dry-run servos, no sleeping")
    ap.add_argument("--dry-run", action="store_true", help="real IMU, servos disabled")
    ap.add_argument("--log", type=Path)
    sys.exit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
