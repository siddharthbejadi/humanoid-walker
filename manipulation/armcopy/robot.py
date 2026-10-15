"""Right arm of the real robot (PCA 0x40: shoulder ch3 = joint 13, arm ch2 = 14, elbow ch1 = 15, gripper ch0 = 18) driven from the laptop.

Safety design (read before using):
  * dry-run by default (live=False): nothing is sent.
  * every pulse is clamped to the per-joint [min_us, max_us] in robot_arm_setup.json (measure them with arm_jog.py); without that file the
    walking calibration limits are used (tiny, so the arm barely moves).
  * slew-limited: pulses move at most `max_us_per_s` toward the target, so a bad detection cannot snap the arm.
  * a 50 Hz sender thread keeps the STM32 watchdog (100 ms) fed; all other joints are held at their calibration neutral, as in robot_host.py --only.
  * go_home() eases to the home pulses and keeps holding; relax() cuts torque (the arm will drop).
"""
import json, os, sys, threading, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deploy"))            # serial_servo_link.py lives in the locomotion deploy folder (not duplicated)
from serial_servo_link import encode_frame

CAL = Path(os.environ.get("WALKER_CALIBRATION", Path(__file__).resolve().parent.parent / "firmware" / "camera_walker" / "servo_calibration_sim.json"))
SETUP = Path(__file__).resolve().parent.parent / "robot_arm_setup.json"
IDX = {"shoulder": 13, "arm": 14, "elbow": 15, "grip": 18}


def load_setup():
    cal = json.loads(CAL.read_text()); setup = json.loads(SETUP.read_text()) if SETUP.exists() else {}
    out = {}
    for name, i in IDX.items():
        c = cal[i]; s = setup.get(name, {})
        lo, hi = s.get("min_us", c["min_us"]), s.get("max_us", c["max_us"])
        home = float(np.clip(s.get("home_us", c["neutral_us"]), min(lo, hi), max(lo, hi)))     # the firmware clamps to the same window, e.g. gripper neutral 1500 is outside 2150-2200
        out[name] = dict(idx=i, home_us=home, min_us=lo, max_us=hi,
                         sign=s.get("sign", c["sign"]), us_per_rad=s.get("us_per_rad", c["us_per_rad"]), measured=name in setup)
    return cal, out


class RightArm:
    def __init__(self, port=None, live=False, max_us_per_s=200.0, rate_hz=50.0):
        self.cal, self.j = load_setup(); self.live, self.max_us_per_s, self.dt = live, max_us_per_s, 1.0 / rate_hz
        self.base = [int(c["neutral_us"]) for c in self.cal]
        self.cur = {n: float(v["home_us"]) for n, v in self.j.items()}; self.tgt = dict(self.cur)
        self.ser = None; self.seq = 0; self.stop = False; self.rx = b""; self.last_msg = ""; self.lock = threading.Lock(); self.relaxed = False; self.sent = 0
        if live:
            import serial; self.ser = serial.Serial(port, 115200, timeout=0.0, write_timeout=0.05)
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start()

    def pulse_for(self, name, angle_rad):
        j = self.j[name]; us = j["home_us"] + j["sign"] * j["us_per_rad"] * float(angle_rad)
        return float(np.clip(us, min(j["min_us"], j["max_us"]), max(j["min_us"], j["max_us"])))

    def set_angles(self, shoulder, arm, elbow, grip):
        """sim-convention angles (rad): shoulder + = forward/up, arm + = outward, elbow + = forearm outward, grip 0..1.39 (closed)."""
        with self.lock:
            for n, a in (("shoulder", shoulder), ("arm", arm), ("elbow", elbow), ("grip", grip)): self.tgt[n] = self.pulse_for(n, a)

    def set_pulse(self, name, us):                       # used by arm_jog (still clamped to hard limits 500..2500 only)
        with self.lock: self.tgt[name] = float(np.clip(us, 500, 2500))

    def go_home(self):
        with self.lock: self.tgt = {n: float(v["home_us"]) for n, v in self.j.items()}

    def relax(self):
        self.relaxed = True
        if self.ser:
            for _ in range(5): self.seq += 1; self.ser.write(encode_frame(self.seq, self.base, relax=True))

    def _run(self):
        t_next = time.time()
        while not self.stop:
            with self.lock: tgt = dict(self.tgt)
            for n in self.cur:
                d = tgt[n] - self.cur[n]; step = self.max_us_per_s * self.dt; self.cur[n] += float(np.clip(d, -step, step))
            if not self.relaxed:
                pulses = list(self.base)
                for n, v in self.j.items(): pulses[v["idx"]] = int(round(self.cur[n]))
                self.seq += 1; self.sent += 1
                if self.ser:
                    try:
                        self.ser.write(encode_frame(self.seq, pulses)); self._rx(self.ser.read(self.ser.in_waiting or 0))       # keep the board's WALKER,... messages, drop the rest
                    except Exception as e: print("serial error:", e); self.stop = True
            t_next += self.dt; time.sleep(max(0.0, t_next - time.time()))

    def _rx(self, data):
        self.rx = (self.rx + data)[-2000:]
        while b"\n" in self.rx:
            line, self.rx = self.rx.split(b"\n", 1); t = line.decode("ascii", "ignore").strip()
            if t.startswith("WALKER,"): self.last_msg = t[:70]

    def close(self, relax=False):
        self.go_home(); t0 = time.time()
        while time.time() - t0 < 6 and any(abs(self.cur[n] - self.j[n]["home_us"]) > 2 for n in self.cur): time.sleep(0.05)
        if relax: self.relax()
        self.stop = True; self.th.join(1.0)
        if self.ser: self.ser.close()


if __name__ == "__main__":                              # self-check, dry-run (nothing is sent)
    a = RightArm(live=False, max_us_per_s=200)
    a.set_angles(1.5, 1.5, 1.5, 1.39); time.sleep(0.5)
    for n, v in a.j.items(): assert min(v["min_us"], v["max_us"]) - 1 <= a.cur[n] <= max(v["min_us"], v["max_us"]) + 1, (n, a.cur[n])
    assert all(abs(a.cur[n] - a.j[n]["home_us"]) <= 200 * 0.55 + 2 for n in a.cur), "slew limit violated"
    print("dry-run self-check OK; limits in use:", {n: (v["min_us"], v["max_us"], "measured" if v["measured"] else "walking-default") for n, v in a.j.items()}); a.stop = True
