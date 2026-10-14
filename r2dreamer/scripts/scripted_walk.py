"""Scripted quasi-static walk for the humanoid (design + open-loop test in MuJoCo, no policy).

Why: the vendor gait pattern cannot make a step in this simulator (it tips over at ~5 s, long before
its stepping section; feet never leave the floor), so the learner had nothing to copy. This builds a
walk that DOES step, by inverse kinematics over the torso position and both legs' joints:

  * the stance foot stays exactly where it is (flat sole),
  * the whole-body centre of mass is moved over the stance foot's centre (weight shift),
  * the swing foot lifts, moves `step` forward on a flat sole, and lowers,
  * then the weight moves to between the feet and the other foot swings.

Everything is solved on a dense time grid, so the motor targets are one smooth [T, 19] table.
Forward is world -y, sideways world +x (as in the rest of the project).

    python scripts/scripted_walk.py --steps 6           # build, simulate, print the report
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
FORWARD = np.array([0., -1.])
LEG = {"left": slice(0, 5), "right": slice(5, 10)}        # actuator indices 0..9 = the two legs
DT = .02                                                  # control period (action_repeat 10 x 2 ms)


def load(scene=None):
    derived = ROOT / "envs/mjx_walking/assets/humanoid_rubber_2mm_servo_v2_kp40_mjx_contacts_18kgcm.xml"   # 1.77 N*m servos, as in training
    path = Path(scene) if scene else (derived if derived.exists() else ROOT / "envs/mjx_walking/assets/humanoid_rubber_2mm_servo_v2_kp40.xml")
    m = mujoco.MjModel.from_xml_path(str(path))
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "standing_start")
    return m, key


class Kin:
    """Whole-body kinematics for IK: variables = torso xyz + 10 leg joints; torso stays upright."""

    def __init__(self, m, key):
        self.m, self.key = m, key
        self.d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, self.d, key)
        self.base = self.d.qpos.copy()
        self.aq = m.jnt_qposadr[m.actuator_trnid[:, 0]]
        self.lo, self.hi = m.actuator_ctrlrange[:10, 0], m.actuator_ctrlrange[:10, 1]
        self.pads = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot_rubber_pad_2mm_candidate") for s in LEG}
        self.pad_pts = {}
        for s, g in self.pads.items():
            a = m.mesh_vertadr[m.geom_dataid[g]]
            v = m.mesh_vert[a:a + m.mesh_vertnum[m.geom_dataid[g]]]
            bottom = v[:len(v) // 2]                       # pad mesh: bottom face first, then top face
            f, l = bottom[:, 1], bottom[:, 0]              # local frame; pick 4 spread-out sole points
            self.pad_pts[s] = bottom[[int(np.argmin(f)), int(np.argmax(f)), int(np.argmin(l)), int(np.argmax(l))]]
        self.x0 = np.r_[self.base[:3], self.base[self.aq[:10]]]
        self.set(self.x0)
        self.sole0 = {s: self.sole(s) for s in LEG}
        self.com0 = self.d.subtree_com[0, :2].copy()
        self.pad_centre0 = {s: self.sole0[s][:, :2].mean(0) for s in LEG}
        self.mass = float(m.body_mass.sum())

    def set(self, x):
        self.d.qpos[:] = self.base
        self.d.qpos[:3] = x[:3]
        self.d.qpos[self.aq[:10]] = x[3:]
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def sole(self, s):
        g = self.pads[s]
        return self.pad_pts[s] @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]

    def solve(self, x_init, shift, com_xy, w_com=6., w_reg=.02, w_z=.2, z_drop=0.):
        """shift: {foot: (dx, dy, dz)} translation of that foot's flat sole from standing; com_xy target."""
        target = {s: self.sole0[s] + np.asarray(shift.get(s, (0., 0., 0.))) for s in LEG}

        def res(x):
            self.set(x)
            r = [10. * (self.sole(s) - target[s]).ravel() for s in LEG]
            r.append(w_com * (self.d.subtree_com[0, :2] - com_xy))
            r.append(w_reg * (x[3:] - self.x0[3:]))
            r.append(np.array([w_z * (x[2] - (self.x0[2] - z_drop))]))
            return np.concatenate(r)

        lo = np.r_[[-np.inf] * 3, self.lo]
        hi = np.r_[[np.inf] * 3, self.hi]
        sol = least_squares(res, np.clip(x_init, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), xtol=1e-10, ftol=1e-10, gtol=1e-10)
        self.set(sol.x)
        err = max(np.abs(self.sole(s) - target[s]).max() for s in LEG)
        return sol.x, err, self.d.subtree_com[0, :2] - com_xy


def smooth(u):
    u = np.clip(u, 0., 1.)
    return u * u * u * (10. - 15. * u + 6. * u * u)            # minimum-jerk blend 0 -> 1


def build_path(kin, n_steps=6, step=.05, lift=.03, t_shift=1.6, t_lift=.8, t_swing=1.2, t_lower=1.2, t_settle=2.0,
               first="right", com_bias=(0., 0.), start_hold=1.0, preload=.002, z_drop=0., settle_to=.5, narrow=.015, t_narrow=2.):
    """Returns the leg-joint target table [T, 10] (+ torso z) and phase labels, on a DT grid.
    Foot targets are expressed as translations of each sole from its standing position."""
    foot_pos = {s: np.zeros(3) for s in LEG}                   # current sole translation (x side, y fwd(-), z up)
    com = kin.com0.copy()                                      # whole-body CoM target (world xy)
    seq, labels = [], []
    x = kin.x0.copy()
    fails = []

    def emit(dur, foot_to, com_to, label, drop=None):
        nonlocal x, com
        drop = z_drop if drop is None else drop
        n = max(int(round(dur / DT)), 1)
        f0 = {s: foot_pos[s].copy() for s in LEG}
        c0 = com.copy()
        for i in range(1, n + 1):
            u = smooth(i / n)
            shift = {s: f0[s] + u * (np.asarray(foot_to.get(s, f0[s])) - f0[s]) for s in LEG}
            cxy = c0 + u * (np.asarray(com_to) - c0)
            x, err, cerr = kin.solve(x, shift, cxy, z_drop=drop)
            if err > 2e-3 or np.abs(cerr).max() > 3e-3:
                fails.append((label, i, round(err * 1000, 2), np.round(cerr * 1000, 2)))
            seq.append(x.copy()); labels.append(label)
        for s in LEG:
            if s in foot_to: foot_pos[s] = np.asarray(foot_to[s], float)
        com = np.asarray(com_to, float)

    emit(start_hold, {}, kin.com0, "hold", drop=0.)
    if narrow > 0:                                             # bring the feet closer together first
        emit(t_narrow, {"left": np.array([-narrow, 0., 0.]), "right": np.array([narrow, 0., 0.])}, kin.com0, "narrow stance")
    swing_order = [first, "left" if first == "right" else "right"]
    for k in range(n_steps):
        swing = swing_order[k % 2]; stance = "left" if swing == "right" else "right"
        stance_xy = kin.pad_centre0[stance] + foot_pos[stance][:2]
        emit(t_shift, {}, stance_xy + np.asarray(com_bias), f"{k}: shift over {stance}")
        up = foot_pos[swing] + np.array([0., 0., lift])
        emit(t_lift, {swing: up}, com, f"{k}: lift {swing}")
        fwd = up + np.array([0., -step, 0.]) * (1. if k > 0 else 1.)
        emit(t_swing, {swing: fwd}, com, f"{k}: swing {swing}")
        down = fwd - np.array([0., 0., lift + preload])      # press slightly into the floor, like the standing pose
        emit(t_lower, {swing: down}, com, f"{k}: lower {swing}")
        mid = (settle_to * (kin.pad_centre0[swing] + foot_pos[swing][:2])
               + (1. - settle_to) * (kin.pad_centre0[stance] + foot_pos[stance][:2]))
        mid = mid + np.array([0., 0.])
        emit(t_settle, {}, mid, f"{k}: settle between feet")
    return np.array(seq), labels, fails


def simulate(m, key, kin, table, speed_check=True, slew=2.):
    """Open-loop playback: leg targets from `table`, everything else holds standing values."""
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, key)
    aq = kin.aq
    ctrl = d.qpos[aq].copy()
    stand = ctrl.copy()
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    rec = dict(t=[], tilt=[], z=[], fwd=[], lift_l=[], lift_r=[], load_l=[], load_r=[], com_f=[])
    fell = None
    geoms = {s: kin.pads[s] for s in LEG}
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for k in range(len(table)):
        tgt = stand.copy()
        tgt[:10] = table[k][3:]
        ctrl = np.clip(tgt, ctrl - slew * DT, ctrl + slew * DT)
        d.ctrl[:] = ctrl
        for _ in range(int(round(DT / m.opt.timestep))):
            mujoco.mj_step(m, d)
        tilt = float(np.arccos(np.clip(d.xmat[base].reshape(3, 3)[2, 2], -1, 1)))
        low = {s: float(((kin.pad_pts[s] @ d.geom_xmat[g].reshape(3, 3).T) + d.geom_xpos[g])[:, 2].min()) for s, g in geoms.items()}
        load = {s: 0. for s in LEG}
        f6 = np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]
            for s, g in geoms.items():
                if (con.geom1 == g and con.geom2 == floor) or (con.geom2 == g and con.geom1 == floor):
                    mujoco.mj_contactForce(m, d, c, f6)
                    load[s] += f6[0]
        rec["t"].append((k + 1) * DT); rec["tilt"].append(tilt); rec["z"].append(float(d.qpos[2]))
        rec["fwd"].append(float(-d.qpos[1])); rec["lift_l"].append(low["left"]); rec["lift_r"].append(low["right"])
        rec["load_l"].append(load["left"]); rec["load_r"].append(load["right"])
        rec["com_f"].append(float(-d.subtree_com[0, 1]))
        if fell is None and (tilt > .45 or d.qpos[2] < .2):
            fell = (k + 1) * DT
            break
    return {k: np.array(v) for k, v in rec.items()}, fell


def export_table(kin, path, n_steps=6, **kw):
    """Save the reference as a start-up part + a repeating 2-step cycle (controls for all 19 actuators)."""
    table, labels, fails = build_path(kin, n_steps, **kw)
    ends = {}
    for i, l in enumerate(labels):
        if "settle between feet" in l:
            ends[int(l.split(":")[0])] = i
    start, period = ends[1] + 1, ends[3] - ends[1]            # cycle = steps 2 and 3 (state after step 1 repeats)
    full = np.tile(kin.x0[3:][:0], 0)
    ctrl = np.tile(kin.base[kin.aq], (len(table), 1))
    ctrl[:, :10] = table[:, 3:]
    np.savez(path, ctrl=ctrl.astype(np.float32), cycle_start=start, cycle_len=period, dt=DT, torso=table[:, :3].astype(np.float32))
    return start, period, len(fails)


def export_phase_pool(m, key, kin, gait_npz, path, every=25):
    """Record body states along the scripted walk (open loop, CPU physics) so training episodes can start at ANY
    phase of it (reference state initialisation). Entry 0 is a dummy 'standing' slot (category 0 uses the
    standing keyframe); entries 1.. are phase states with the gait index to resume at."""
    g = np.load(gait_npz)
    ctrl_tab, start, period = g["ctrl"], int(g["cycle_start"]), int(g["cycle_len"])
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, key)
    ctrl = d.qpos[kin.aq].copy()
    qp, qv, cc, ph = [d.qpos.copy()], [d.qvel.copy()], [ctrl.copy()], [0]
    n = start + period
    for k in range(n):
        ctrl = np.clip(ctrl_tab[k], ctrl - 2. * DT, ctrl + 2. * DT)
        d.ctrl[:] = ctrl
        for _ in range(int(round(DT / m.opt.timestep))):
            mujoco.mj_step(m, d)
        if (k + 1) % every == 0:
            tilt = float(np.arccos(np.clip(d.xmat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")].reshape(3, 3)[2, 2], -1, 1)))
            if tilt > .2:
                raise RuntimeError(f"open-loop playback tilted {np.degrees(tilt):.0f} deg at step {k + 1}; pool not trustworthy")
            qp.append(d.qpos.copy()); qv.append(d.qvel.copy()); cc.append(ctrl.copy()); ph.append(k + 1)
    np.savez(path, qpos=np.asarray(qp, np.float32), qvel=np.asarray(qv, np.float32), ctrl=np.asarray(cc, np.float32),
             phase_offset=np.asarray(ph, np.int32), n=len(qp) - 1)
    return len(qp) - 1


def view(m, key, kin, table, speed=1.0):
    """Play the walk in the interactive MuJoCo viewer (real time)."""
    import time
    import mujoco.viewer
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, key)
    ctrl = d.qpos[kin.aq].copy()
    stand = ctrl.copy()
    with mujoco.viewer.launch_passive(m, d) as v:
        for k in range(len(table)):
            if not v.is_running():
                break
            t0 = time.time()
            tgt = stand.copy(); tgt[:10] = table[k][3:]
            ctrl = np.clip(tgt, ctrl - 2. * DT, ctrl + 2. * DT)
            d.ctrl[:] = ctrl
            for _ in range(int(round(DT / m.opt.timestep))):
                mujoco.mj_step(m, d)
            v.sync()
            time.sleep(max(0., DT / speed - (time.time() - t0)))
        while v.is_running():
            time.sleep(.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--step", type=float, default=.05)
    ap.add_argument("--lift", type=float, default=.03)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--view", action="store_true", help="play the walk in the interactive viewer")
    ap.add_argument("--export", default=None, help="write the reference table (.npz) for training")
    ap.add_argument("--export-pool", default=None, help="write the phase start-state pool (.npz); needs --export-from")
    ap.add_argument("--export-from", default=None, help="the reference table .npz to play for --export-pool")
    a = ap.parse_args()
    m, key = load(a.scene)
    kin = Kin(m, key)
    table, labels, fails = build_path(kin, a.steps, a.step, a.lift)
    print(f"path: {len(table)} control steps = {len(table) * DT:.1f} s; IK misses: {len(fails)}", fails[:3])
    if a.export:
        print("exported", a.export, "cycle start/len (control steps) and IK misses:", export_table(kin, a.export, max(a.steps, 6), step=a.step, lift=a.lift))
    if a.export_pool:
        print("phase pool states:", export_phase_pool(m, key, kin, a.export_from or a.export, a.export_pool))
        return
    if a.view:
        view(m, key, kin, table)
        return
    rec, fell = simulate(m, key, kin, table)
    print("fell at", fell, "s" if fell else "(stayed up the whole script)")
    print("forward progress of torso: %.1f cm; max tilt %.1f deg; highest sole lift L/R %.1f/%.1f cm" % (
        (rec["fwd"][-1] - rec["fwd"][0]) * 100, np.degrees(rec["tilt"].max()), rec["lift_l"].max() * 100, rec["lift_r"].max() * 100))


if __name__ == "__main__":
    main()
