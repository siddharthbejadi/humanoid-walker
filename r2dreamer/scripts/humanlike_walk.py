"""Human-like walk for THIS robot: planner from the user's files.zip (ZMP preview control, footsteps, swing profile),
joint angles from our own numeric IK (this robot has 5-joint legs: hip roll, three forward/back joints, ankle roll;
no hip yaw and no ankle pitch, so the zip's analytic 6-joint IK cannot be used).

    python scripts/humanlike_walk.py --step-len 0.025 --step-time 0.65        # build + simulate + report
Forward is world -y, sideways world +x (planner frame: x forward, y left -> world (X, Y) = (y, -x)).
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import scripted_walk as sw                      # Kin (kinematics), simulate (CPU playback), load
from zip_humanlike_gait import GaitParams, RobotGeom, generate

FOOT_FWD0 = 0.017           # standing sole centre is 1.7 cm ahead of the torso origin
LAT0 = 0.003                # torso origin sits 0.3 cm off the midline between the feet


def robot_geom(kin):
    """RobotGeom measured from this robot (hip pitch->ankle 17.2 cm split in two, hip roll joints 8.9 cm apart)."""
    d = kin.d
    com = d.subtree_com[0]
    return RobotGeom(thigh=.086, shank=.086, ankle_h=.017, hip_y=.0445, hip_drop=.111, foot_half_len=.0615,
                     com_offset_x=float(-(com[1] - d.qpos[1])) * -1., com_above_pelvis=float(com[2] - d.qpos[2]))


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def solve(kin, x0, feet, com_xy, w_com=6., w_reg=.02, w_z=.2, z_drop=.02):
    """feet: {side: (dX, dY, dZ, pitch)} sole-centre translation from standing + pitch about it (+ = toe down)."""
    targets = {}
    for s, (dx, dy, dz, pitch) in feet.items():
        c0 = kin.sole0[s].mean(0)
        targets[s] = c0 + np.array([dx, dy, dz]) + (kin.sole0[s] - c0) @ rx(pitch).T

    def res(x):
        kin.set(x)
        r = [10. * (kin.sole(s) - targets[s]).ravel() for s in targets]
        r.append(w_com * (kin.d.subtree_com[0, :2] - com_xy))
        r.append(w_reg * (x[3:] - kin.x0[3:]))
        r.append(np.array([w_z * (x[2] - (kin.x0[2] - z_drop))]))
        return np.concatenate(r)

    lo = np.r_[[-np.inf] * 3, kin.lo]; hi = np.r_[[np.inf] * 3, kin.hi]
    sol = least_squares(res, np.clip(x0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), xtol=1e-9, ftol=1e-9, gtol=1e-9)
    kin.set(sol.x)
    err = max(np.abs(kin.sole(s) - targets[s]).max() for s in targets)
    return sol.x, err, np.abs(kin.d.subtree_com[0, :2] - com_xy).max()


def build(kin, step_len=.025, step_time=.65, n_steps=10, ds_ratio=.2, clearance=.02, toe_pitch=.15, z_drop=.02,
          t_transition=1.5, t_start=1.0, t_end=1.5, zmp_inset=.01, step_width=.089, stride_bias=0., **extra):
    geom = robot_geom(kin)
    p = GaitParams(step_len=step_len, step_time=step_time, n_steps=n_steps, ds_ratio=ds_ratio, clearance=clearance,
                   toe_pitch=toe_pitch, step_width=step_width, pelvis_yaw=0., torso_lean=0., arm_swing=0., arm_abduct=0.,
                   elbow_bias=0., t_start=t_start, t_end=t_end, zmp_inset=zmp_inset, **extra)
    tr = generate(geom, p)
    t, step = tr["t"], int(round(sw.DT / p.dt))
    idx = np.arange(0, len(t), step)
    com_plan = tr["com"][idx]                      # (x forward, y left) of the CoM in the planner frame
    rows, fails = [], []
    x = kin.x0.copy()
    for k in idx:
        feet = {}
        for s in ("left", "right"):
            f = tr["foot"][s[0].upper()][k]
            scale = 1. + stride_bias * (1. if s == "left" else -1.)     # differential stride = steering (no hip yaw joint)
            feet[s] = (0., -f[0] * scale, f[2], float(tr["pitch"][s[0].upper()][k]))
        com_xy = np.array([com_plan[list(idx).index(k)][1] + LAT0, -(com_plan[list(idx).index(k)][0] + FOOT_FWD0)]) + 0.
        x, err, cerr = solve(kin, x, feet, com_xy, z_drop=z_drop)
        if err > 3e-3 or cerr > 4e-3:
            fails.append((round(t[k], 2), round(err * 1000, 1), round(cerr * 1000, 1)))
        rows.append(x.copy())
    rows = np.array(rows)
    # standing keyframe -> first planned pose (quasi-static, minimum jerk)
    n_tr = int(t_transition / sw.DT)
    blend = np.array([sw.smooth(i / n_tr) for i in range(1, n_tr + 1)])[:, None]
    trans = kin.x0 + blend * (rows[0] - kin.x0)
    return np.vstack([trans, rows]), tr, fails, n_tr


def arm_signal(tr, n_tr, step_len):
    """d(t) in [-1, 1]: +1 when the left foot is furthest ahead (zip's arm-swing phase), aligned with the table rows."""
    step = int(round(sw.DT / .01))
    d = np.clip((tr["foot"]["L"][:, 0] - tr["foot"]["R"][:, 0]) / max(step_len, 1e-6), -1, 1)[::step]
    return np.r_[np.zeros(n_tr), d]


# ----------------------------------------------------------------------------- closed-loop playback (tilt stabiliser)
def play(m, key, kin, table, k_roll=1.0, d_roll=.04, k_pitch=0., d_pitch=0., hip_roll_sign=-1., limit=.25, slew=8., stop_on_fall=True,
         on_step=None, arm=None, data=None):
    """Play the table on the CPU model with a torso-tilt stabiliser (the zip's IMU idea): sideways tilt -> ankle + hip roll,
    forward/back tilt -> the hip pitch joints (the foot is kept flat by the next pitch joint). Ankle-roll signs are
    mirrored because the two ankle axes are mirrored. Returns (record dict, time of fall or None)."""
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    d = mujoco.MjData(m) if data is None else data           # pass the viewer's own MjData to see the walk in the window
    mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    aq = kin.aq
    ctrl = d.qpos[aq].copy()
    stand = ctrl.copy()
    prev = np.zeros(2); filt = np.zeros(2)
    rec = dict(t=[], tilt=[], roll=[], pitch=[], z=[], fwd=[], side=[])
    fell = None
    for k in range(len(table)):
        tgt = stand.copy()
        tgt[:10] = table[k][3:]
        R = d.xmat[base].reshape(3, 3)
        roll, pitch = float(np.arcsin(np.clip(R[0, 2], -1, 1))), float(np.arcsin(np.clip(R[1, 2], -1, 1)))
        err = np.array([roll, pitch])
        rate = (err - prev) / sw.DT; prev = err
        filt = .8 * filt + .2 * rate
        c_roll = np.clip(k_roll * roll + d_roll * filt[0], -limit, limit)
        c_pitch = np.clip(k_pitch * pitch + d_pitch * filt[1], -limit, limit)
        tgt[0] += hip_roll_sign * c_roll; tgt[5] += hip_roll_sign * c_roll
        tgt[4] += c_roll; tgt[9] -= c_roll
        for hp, th in ((1, 2), (6, 7)):                       # hip pitch +, next pitch joint - : leg shifts, foot stays flat
            tgt[hp] += c_pitch; tgt[th] -= c_pitch
        if arm is not None:                                   # arm = (amplitude rad, sign, d(t) per row); shoulder axes are mirrored
            a_amp, a_sign, a_d = arm
            tgt[10] += -a_sign * a_amp * a_d[min(k, len(a_d) - 1)]; tgt[13] += -a_sign * a_amp * a_d[min(k, len(a_d) - 1)]
        ctrl = np.clip(tgt, ctrl - slew * sw.DT, ctrl + slew * sw.DT)
        d.ctrl[:] = ctrl
        for _ in range(int(round(sw.DT / m.opt.timestep))):
            mujoco.mj_step(m, d)
        R = d.xmat[base].reshape(3, 3)
        tilt = float(np.arccos(np.clip(R[2, 2], -1, 1)))
        rec["t"].append((k + 1) * sw.DT); rec["tilt"].append(tilt); rec["roll"].append(roll); rec["pitch"].append(pitch)
        rec["z"].append(float(d.qpos[2])); rec["fwd"].append(float(-d.qpos[1])); rec["side"].append(float(d.qpos[0]))
        if on_step:
            on_step(k, d)
        if fell is None and (tilt > .45 or d.qpos[2] < .2):
            fell = (k + 1) * sw.DT
            if stop_on_fall:
                break
    return {k: np.array(v) for k, v in rec.items()}, fell


BEST = dict(step_len=.025, step_time=.40, ds_ratio=.2, zmp_inset=.02, toe_pitch=0.)   # 5 cm per foot per step; 84 s, 6.2 cm/s, straight (0.40 s/step)
STAB = dict(k_roll=1.0, d_roll=.04)                                                      # tilt stabiliser gains used with BEST


def render_video(m, key, kin, table, tr, n_tr, path, arm, fps=25, size=(480, 360)):
    """Record the walk (closed-loop playback) from two tracking cameras, side by side, to an mp4."""
    import subprocess
    import imageio_ffmpeg
    renderer = mujoco.Renderer(m, size[1], size[0])
    cams = []
    for az, el, dist in ((140., -12., .85), (0., -3., .8)):
        c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FREE; c.azimuth, c.elevation, c.distance = az, el, dist
        cams.append(c)
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{size[0] * 2}x{size[1]}", "-r", str(fps), "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "23", path], stdin=subprocess.PIPE)
    every = max(int(round(1. / (fps * sw.DT))), 1)

    def on_step(k, d):
        if k % every:
            return
        views = []
        for c in cams:
            c.lookat[:] = [d.qpos[0], d.qpos[1], .17]
            renderer.update_scene(d, camera=c)
            views.append(renderer.render().copy())
        ff.stdin.write(np.concatenate(views, axis=1).tobytes())

    rec, fell = play(m, key, kin, table, on_step=on_step, arm=arm, stop_on_fall=True)
    ff.stdin.close(); ff.wait()
    return rec, fell


def view(m, key, kin, table, arm, speed=1.0):
    """Play the closed-loop walk in the interactive MuJoCo viewer (real time), camera following the robot.
    The viewer and the walk MUST share one MjData, otherwise the window shows an untouched copy of the robot."""
    import time
    import mujoco.viewer
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.distance, v.cam.azimuth, v.cam.elevation = .9, 140., -12.
        t0 = time.time()
        print("Walking. Close the window to stop.", flush=True)

        def on_step(k, dd):
            if not v.is_running():
                raise KeyboardInterrupt
            v.cam.lookat[:] = [dd.qpos[0], dd.qpos[1], .17]
            v.sync()
            time.sleep(max(0., t0 + (k + 1) * sw.DT / speed - time.time()))
        try:
            rec, fell = play(m, key, kin, table, on_step=on_step, arm=arm, stop_on_fall=True, data=d)
            print("fell at %.1f s" % fell if fell else "walk finished (%.1f s)" % (len(rec["t"]) * sw.DT), flush=True)
            while v.is_running():
                v.sync(); time.sleep(.05)
        except KeyboardInterrupt:
            pass


def export(m, key, kin, table_path, pool_path, arm_amp=.6, n_steps=60, every=5):
    """Training files: (1) reference joint targets for all 19 actuators, start-up + a repeating 2-step cycle (open loop,
    WITHOUT the stabiliser: the environment adds that itself); (2) body states recorded from the closed-loop walk at many
    phases, so training episodes can start anywhere in the walk."""
    table, tr, fails, n_tr = build(kin, n_steps=n_steps, **BEST)
    d_sig = arm_signal(tr, n_tr, BEST["step_len"])
    ctrl = np.tile(kin.base[kin.aq], (len(table), 1))
    ctrl[:, :10] = table[:, 3:]
    ctrl[:, 10] += arm_amp * d_sig[:len(ctrl)]                       # arm = (amp, -1, d): both shoulders += amp * d
    ctrl[:, 13] += arm_amp * d_sig[:len(ctrl)]
    period = int(round(2 * BEST["step_time"] / sw.DT))
    start = n_tr + int(round((1.0 + 4 * BEST["step_time"]) / sw.DT))
    err = np.abs(ctrl[start:start + period] - ctrl[start + period:start + 2 * period]).max()
    err2 = np.abs(ctrl[start + period:start + 2 * period] - ctrl[start + 2 * period:start + 3 * period]).max()
    print(f"cycle: start {start} ({start * sw.DT:.1f} s), period {period} steps ({period * sw.DT:.2f} s); repeat error {np.degrees(err):.2f} / {np.degrees(err2):.2f} deg; IK misses {len(fails)}")
    np.savez(table_path, ctrl=ctrl.astype(np.float32), cycle_start=start, cycle_len=period, dt=sw.DT)
    states = {"qpos": [], "qvel": [], "ctrl": [], "phase": []}

    def on_step(k, d):
        if (k + 1) % every == 0 and k + 1 <= start + 2 * period:
            states["qpos"].append(d.qpos.copy()); states["qvel"].append(d.qvel.copy()); states["ctrl"].append(d.ctrl.copy()); states["phase"].append(k + 1)

    arm = (arm_amp, -1, d_sig)
    rec, fell = play(m, key, kin, table, arm=arm, on_step=on_step)
    if fell and fell < (start + 2 * period) * sw.DT:
        raise RuntimeError(f"closed-loop walk fell at {fell:.1f} s, before the pool was recorded")
    d0 = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d0, key)
    q = [d0.qpos.copy()] + states["qpos"]; v = [d0.qvel.copy()] + states["qvel"]; c = [d0.ctrl.copy()] + states["ctrl"]; ph = [0] + states["phase"]
    np.savez(pool_path, qpos=np.asarray(q, np.float32), qvel=np.asarray(v, np.float32), ctrl=np.asarray(c, np.float32),
             phase_offset=np.asarray(ph, np.int32), n=len(q) - 1)
    print(f"table {ctrl.shape} -> {table_path}; phase pool {len(q) - 1} states -> {pool_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-training", nargs=2, metavar=("TABLE_NPZ", "POOL_NPZ"), help="write the training reference and phase pool")
    ap.add_argument("--step-len", type=float, default=BEST["step_len"], help="half stride [m]: each foot advances 2x this per own step")
    ap.add_argument("--step-time", type=float, default=BEST["step_time"])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--arm", type=float, default=.6, help="arm swing amplitude [rad]")
    ap.add_argument("--video", default=None, help="write an mp4 of the walk (needs MUJOCO_GL=glfw on this laptop)")
    ap.add_argument("--view", action="store_true", help="play in the interactive viewer")
    ap.add_argument("--export", default=None, help="write the reference table (.npz) for training")
    a = ap.parse_args()
    m, key = sw.load()
    kin = sw.Kin(m, key)
    if a.export_training:
        export(m, key, kin, *a.export_training)
        return
    plan = dict(BEST, step_len=a.step_len, step_time=a.step_time)
    table, tr, fails, n_tr = build(kin, n_steps=a.steps, **plan)
    arm = (a.arm, -1, arm_signal(tr, n_tr, a.step_len))
    print(f"table {len(table)} control steps = {len(table) * sw.DT:.1f} s; IK misses {len(fails)}")
    if a.view:
        view(m, key, kin, table, arm)
        return
    if a.video:
        rec, fell = render_video(m, key, kin, table, tr, n_tr, a.video, arm)
    else:
        rec, fell = play(m, key, kin, table, arm=arm)
    n = len(rec["t"])
    print("FELL at %.2f s" % fell if fell else "stayed up the whole %.1f s" % (n * sw.DT))
    print("torso forward %.1f cm in %.1f s (%.2f cm/s); sideways drift %+.1f cm; max tilt %.1f deg" % (
        rec["fwd"][-1] * 100, n * sw.DT, rec["fwd"][-1] / (n * sw.DT) * 100, rec["side"][-1] * 100, np.degrees(rec["tilt"].max())))


if __name__ == "__main__":
    main()
