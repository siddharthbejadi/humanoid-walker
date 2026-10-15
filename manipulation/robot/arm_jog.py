"""Slow, manual jog of the 4 right-arm servos to find each joint's SAFE range and direction. Writes robot_arm_setup.json.
    python robot/arm_jog.py --port COM3        (robot hanging/on a stand, arm free, a hand on the power switch)
Moves are slew-limited to 200 us/s.  Jog a joint to just before it touches anything, mark that end.  Nothing is saved until you press S."""
import argparse, json, msvcrt, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armcopy import robot as robot_arm
from armcopy.robot import RightArm, SETUP

ap = argparse.ArgumentParser(); ap.add_argument("--port", required=True); a = ap.parse_args()
POS = {"shoulder": "arm swings FORWARD / up (toward the front of the robot)", "arm": "arm swings OUTWARD (away from the body, sideways)",
       "elbow": "forearm swings OUTWARD (away from the body, sideways)", "grip": "gripper CLOSES"}
arm = RightArm(a.port, live=True, max_us_per_s=200.0); names = list(robot_arm.IDX); sel = 0
for n in names: arm.tgt[n] = arm.cur[n]
J = arm.j; print(__doc__)
def show():
    n = names[sel]; j = J[n]; print(f"\r[{sel + 1}] {n:8s} now {arm.tgt[n]:6.0f} us | home {j['home_us']:.0f} min {j['min_us']} max {j['max_us']} sign {j['sign']:+d}   (+ means: {POS[n]})  board: {arm.last_msg or '-'}   ", end="", flush=True)
show()
while True:
    if not msvcrt.kbhit(): time.sleep(0.2); show(); continue
    k = msvcrt.getwch(); n = names[sel]; j = J[n]
    if k in "1234": sel = int(k) - 1
    elif k == "a": arm.set_pulse(n, arm.tgt[n] - 10)
    elif k == "d": arm.set_pulse(n, arm.tgt[n] + 10)
    elif k == "A": arm.set_pulse(n, arm.tgt[n] - 40)
    elif k == "D": arm.set_pulse(n, arm.tgt[n] + 40)
    elif k == "h": j["home_us"] = arm.tgt[n]
    elif k == "n": j["min_us"] = int(arm.tgt[n])
    elif k == "x": j["max_us"] = int(arm.tgt[n])
    elif k == "y": j["sign"] = 1 if arm.tgt[n] >= j["home_us"] else -1           # you jogged in the '+' direction described above
    elif k == "g": arm.go_home()
    elif k == "r": arm.relax(); print("\nRELAXED (torque off)")
    elif k == "S":
        out = {m: {"home_us": J[m]["home_us"], "min_us": min(J[m]["min_us"], J[m]["max_us"]), "max_us": max(J[m]["min_us"], J[m]["max_us"]), "sign": J[m]["sign"], "us_per_rad": J[m]["us_per_rad"]} for m in names}
        SETUP.write_text(json.dumps(out, indent=1)); print(f"\nsaved {SETUP}")
    elif k in "qQ\x03": break
    show()
print("\nreturning home..."); arm.close(); print("done")
