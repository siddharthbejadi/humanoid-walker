# Vendored UNCHANGED from the user-supplied files.zip (humanlike_gait.py): ZMP preview-control biped gait planner. Used here only for planning.
"""
humanlike_gait.py — human-like biped walking pattern generator for a 19-DOF servo humanoid.

Pipeline
  1. Footstep planner        : straight-line steps, each foot stays on its own line (no leg crossing)
  2. ZMP reference           : ZMP sits on the stance foot in single support, ramps in double support
  3. ZMP preview control     : Kajita et al. 2003 cart-table model -> smooth CoM (x, y) trajectory
  4. Swing foot              : min-jerk forward motion, real ground clearance, toe-off / heel-strike pitch
  5. Pelvis height           : solved per-sample so the most-extended leg sits at `leg_extension`
                               (straight-ish human knees instead of a permanent crouch); smoothed
  6. Pelvic rotation + arms  : pelvis yaws with the swing leg, arms swing contralaterally
  7. Analytic 6-DOF leg IK   : hip yaw/roll/pitch, knee, ankle pitch/roll (Kajita, Ch.2)

Frame: x forward, y left, z up. Joint sign: right-hand rule about +x (roll), +y (pitch), +z (yaw).
  knee > 0  = flexion,  hip pitch < 0 = thigh forward,  shoulder pitch < 0 = arm forward.
Map these to your MJCF with JOINT_MAP / SIGN in `mujoco_player.py`.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import filtfilt, butter

G = 9.81


# ----------------------------------------------------------------------------- config
@dataclass
class RobotGeom:
    thigh: float = 0.100          # hip-pitch axis -> knee axis  [m]
    shank: float = 0.100          # knee axis -> ankle axis       [m]
    ankle_h: float = 0.030        # ankle axis -> sole            [m]
    hip_y: float = 0.040          # pelvis origin -> hip joint, lateral
    hip_drop: float = 0.030       # pelvis origin -> hip joint, down
    foot_half_len: float = 0.050  # sole half length (toe clearance check)
    com_offset_x: float = 0.0     # CoM ahead of pelvis origin (tune on the real robot)
    com_above_pelvis: float = 0.04


@dataclass
class GaitParams:
    dt: float = 0.01              # 100 Hz control
    step_len: float = 0.06        # stride/2  [m]
    step_width: float = 0.080     # foot centre-to-centre; = 2*hip_y -> feet under hips, like a human
    step_time: float = 0.65       # one step (DS + SS) [s]
    ds_ratio: float = 0.20        # double-support fraction
    n_steps: int = 10
    clearance: float = 0.030      # swing apex height [m]
    toe_pitch: float = 0.20       # max swing-foot pitch (toe-off / heel-strike) [rad]
    leg_extension: float = 0.98   # max hip-ankle distance / (thigh+shank); 1.0 = locked knee
    pelvis_yaw: float = np.deg2rad(4)   # pelvic rotation amplitude
    torso_lean: float = np.deg2rad(2)   # slight forward lean
    arm_swing: float = np.deg2rad(18)   # shoulder pitch amplitude
    elbow_bias: float = np.deg2rad(25)
    arm_abduct: float = np.deg2rad(6)   # keep arms clear of torso
    lift_lead: float = 0.10       # fraction of swing spent lifting before moving forward
    land_lag: float = 0.05
    zmp_inset: float = 0.015      # ZMP target on the inner sole edge (reduces lateral sway)
    t_start: float = 1.0          # quiet stance before walking
    t_end: float = 1.5
    preview_s: float = 1.6


# ----------------------------------------------------------------------------- helpers
def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def min_jerk(s):
    s = np.clip(s, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


# ----------------------------------------------------------------------------- 1-2 footsteps & ZMP ref
def plan_footsteps(p: GaitParams):
    """Returns list of dict(t0, t_ds, t_ss, swing, stance_xy, swing_from, swing_to)."""
    w = p.step_width / 2
    feet = {"L": np.array([0.0, w]), "R": np.array([0.0, -w])}
    T_ds, T_ss = p.step_time * p.ds_ratio, p.step_time * (1 - p.ds_ratio)
    steps, t = [], p.t_start
    swing = "R"
    for k in range(p.n_steps):
        stance = "L" if swing == "R" else "R"
        if k == 0:
            target_x = feet[stance][0] + p.step_len           # first step: half stride
        elif k == p.n_steps - 1:
            target_x = feet[stance][0]                        # last step: feet together
        else:
            target_x = feet[stance][0] + p.step_len
        t_ds = T_ds * (2.0 if k == 0 else 1.0)                # longer initial weight shift
        zmp_xy = feet[stance] - np.array([0, np.sign(feet[stance][1]) * p.zmp_inset])
        steps.append(dict(t0=t, t_ds=t_ds, t_ss=T_ss, swing=swing,
                          stance_xy=zmp_xy,
                          swing_from=feet[swing].copy(),
                          swing_to=np.array([target_x, feet[swing][1]])))
        feet[swing] = steps[-1]["swing_to"].copy()
        t += t_ds + T_ss
        swing = stance
    t_total = t + T_ds + p.t_end
    return steps, t_total, feet


def zmp_reference(p, steps, t_total, final_feet):
    n = int(round(t_total / p.dt))
    t = np.arange(n) * p.dt
    zmp = np.zeros((n, 2))
    prev = np.array([0.0, 0.0])
    zmp[t < p.t_start] = prev
    for st in steps:
        a, b, c = st["t0"], st["t0"] + st["t_ds"], st["t0"] + st["t_ds"] + st["t_ss"]
        m = (t >= a) & (t < b)
        s = min_jerk((t[m] - a) / st["t_ds"])[:, None]
        zmp[m] = prev + s * (st["stance_xy"] - prev)
        zmp[t >= b] = st["stance_xy"]          # forward-fill; later steps overwrite (no FP gaps)
        prev = st["stance_xy"]
    end = (final_feet["L"] + final_feet["R"]) / 2
    a = steps[-1]["t0"] + steps[-1]["t_ds"] + steps[-1]["t_ss"]
    T_ds = p.step_time * p.ds_ratio
    m = (t >= a) & (t < a + T_ds)
    zmp[m] = prev + min_jerk((t[m] - a) / T_ds)[:, None] * (end - prev)
    zmp[t >= a + T_ds] = end
    return t, zmp


# ----------------------------------------------------------------------------- 3 ZMP preview control
def preview_control(zmp_ref, dt, zc, preview_s, Qe=1.0, R=1e-6):
    A = np.array([[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]])
    B = np.array([[dt**3 / 6], [dt**2 / 2], [dt]])
    C = np.array([[1, 0, -zc / G]])
    At = np.block([[np.eye(1), C @ A], [np.zeros((3, 1)), A]])
    Bt = np.vstack([C @ B, B])
    It = np.array([[1.0], [0], [0], [0]])
    Q = np.zeros((4, 4)); Q[0, 0] = Qe
    P = solve_discrete_are(At, Bt, Q, R)
    S = np.linalg.inv(R + Bt.T @ P @ Bt)
    K = S @ Bt.T @ P @ At
    Gi, Gx = K[0, 0], K[0, 1:]
    Ac = At - Bt @ K
    N = int(preview_s / dt)
    Gd = np.zeros(N); Gd[0] = -Gi
    X = -Ac.T @ P @ It
    for j in range(1, N):
        Gd[j] = (S @ Bt.T @ X).item()
        X = Ac.T @ X

    n = len(zmp_ref)
    ref = np.concatenate([zmp_ref, np.repeat(zmp_ref[-1:], N, 0)])
    x = np.array([zmp_ref[0], 0, 0], float)
    com, zmp_out, e = np.zeros((n, 3)), np.zeros(n), 0.0
    for k in range(n):
        p_k = (C @ x).item()
        e += p_k - ref[k]
        u = -Gi * e - Gx @ x - Gd @ ref[k + 1:k + 1 + N]
        com[k] = x
        zmp_out[k] = p_k
        x = A @ x + B[:, 0] * u
    return com, zmp_out


# ----------------------------------------------------------------------------- 7 leg IK / FK
def leg_ik(body_p, body_R, foot_p, foot_R, hip_off, A, B):
    """Kajita IK. foot_p = ankle joint position. Returns [yaw, roll, pitch, knee, ankle_p, ankle_r]."""
    r = foot_R.T @ (body_p + body_R @ hip_off - foot_p)
    C = np.linalg.norm(r)
    c5 = np.clip((C**2 - A**2 - B**2) / (2 * A * B), -1, 1)
    q4 = np.arccos(c5)
    q5a = np.arcsin(np.clip(A / C * np.sin(np.pi - q4), -1, 1))
    q6 = np.arctan2(r[1], r[2])
    if q6 > np.pi / 2: q6 -= np.pi
    elif q6 < -np.pi / 2: q6 += np.pi
    q5 = -np.arctan2(r[0], np.sign(r[2]) * np.hypot(r[1], r[2])) - q5a
    Rm = body_R.T @ foot_R @ Rx(-q6) @ Ry(-q5 - q4)
    q1 = np.arctan2(-Rm[0, 1], Rm[1, 1])
    q2 = np.arctan2(Rm[2, 1], -Rm[0, 1] * np.sin(q1) + Rm[1, 1] * np.cos(q1))
    q3 = np.arctan2(-Rm[2, 0], Rm[2, 2])
    return np.array([q1, q2, q3, q4, q5, q6]), C / (A + B)


def leg_fk(body_p, body_R, q, hip_off, A, B):
    R = body_R @ Rz(q[0]) @ Rx(q[1]) @ Ry(q[2])
    p = body_p + body_R @ hip_off + R @ [0, 0, -A]
    R = R @ Ry(q[3])
    p = p + R @ [0, 0, -B]
    return p, R @ Ry(q[4]) @ Rx(q[5])


# ----------------------------------------------------------------------------- main generator
JOINTS = ["l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ankle_pitch", "l_ankle_roll",
          "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ankle_pitch", "r_ankle_roll",
          "l_shoulder_pitch", "l_shoulder_roll", "l_elbow",
          "r_shoulder_pitch", "r_shoulder_roll", "r_elbow", "head_yaw"]


def generate(geom: RobotGeom = RobotGeom(), p: GaitParams = GaitParams()):
    L = geom.thigh + geom.shank
    steps, t_total, final_feet = plan_footsteps(p)
    t, zmp_ref = zmp_reference(p, steps, t_total, final_feet)
    n = len(t)

    # ---- feet (sole centre position + pitch) for every sample
    foot = {s: np.zeros((n, 3)) for s in "LR"}
    pitch = {s: np.zeros(n) for s in "LR"}
    w = p.step_width / 2
    for s, y in (("L", w), ("R", -w)):
        foot[s][:, 1] = y
    for st in steps:
        s = st["swing"]
        a = st["t0"] + st["t_ds"]
        b = a + st["t_ss"]
        foot[s][t >= b, 0] = st["swing_to"][0]
        m = (t >= a) & (t < b)
        ph = (t[m] - a) / st["t_ss"]
        # lift first, then advance (no toe drag); land before the end (clean heel strike)
        phx = np.clip((ph - p.lift_lead) / (1 - p.lift_lead - p.land_lag), 0, 1)
        foot[s][m, 0] = st["swing_from"][0] + min_jerk(phx) * (st["swing_to"][0] - st["swing_from"][0])
        # zero-velocity quartic bump, slightly front-loaded: z, dz/dt = 0 at lift-off and touch-down
        u = ph**0.85
        foot[s][m, 2] = p.clearance * 16 * u**2 * (1 - u)**2
        moving = abs(st["swing_to"][0] - st["swing_from"][0]) > 1e-4
        # + toe down (toe-off follow-through), - toe up (heel strike prep).
        # Smooth clearance envelope: pitch only grows as the sole gains height, so the
        # toe/heel (z_sole - half_len*|sin th|) never touches the floor, with C1-smooth joints.
        env = min_jerk(foot[s][m, 2] / p.clearance / 0.6)
        pitch[s][m] = (p.toe_pitch if moving else 0.0) * np.sin(2 * np.pi * ph) * env

    # ---- CoM (x,y) from ZMP preview control
    zc = None  # set after height solve; use nominal first
    z_nom = geom.ankle_h + p.leg_extension * L + geom.hip_drop
    zc = z_nom + geom.com_above_pelvis
    comx, zmpx = preview_control(zmp_ref[:, 0], p.dt, zc, p.preview_s)
    comy, zmpy = preview_control(zmp_ref[:, 1], p.dt, zc, p.preview_s)
    pel = np.zeros((n, 3))
    pel[:, 0] = comx[:, 0] - geom.com_offset_x
    pel[:, 1] = comy[:, 0]

    # ---- pelvis yaw (pelvic rotation) & arm swing phase signal
    d = (foot["L"][:, 0] - foot["R"][:, 0]) / max(p.step_len, 1e-6)   # +1 left foot ahead
    d = np.clip(d, -1, 1)
    yaw = -p.pelvis_yaw * d
    body_R = [Rz(yaw[k]) @ Ry(p.torso_lean) for k in range(n)]

    # ---- ankle positions
    ank = {}
    footR = {}
    for s in "LR":
        footR[s] = [Ry(pitch[s][k]) for k in range(n)]
        ank[s] = np.array([foot[s][k] + footR[s][k] @ [0, 0, geom.ankle_h] for k in range(n)])

    # ---- 5: pelvis height so the most-extended leg sits at leg_extension
    zreq = np.full(n, np.inf)
    for s, sy in (("L", 1), ("R", -1)):
        hip = pel + np.array([body_R[k] @ [0, sy * geom.hip_y, -geom.hip_drop] for k in range(n)])
        hz_off = hip[:, 2] - pel[:, 2]
        horiz = np.hypot(hip[:, 0] - ank[s][:, 0], hip[:, 1] - ank[s][:, 1])
        vert = np.sqrt(np.maximum((p.leg_extension * L) ** 2 - horiz**2, 1e-6))
        zreq = np.minimum(zreq, ank[s][:, 2] + vert - hz_off)
    bb, aa = butter(2, 3.0 / (0.5 / p.dt))          # 3 Hz zero-phase smoothing
    pel[:, 2] = filtfilt(bb, aa, zreq) - 0.001
    pel[:, 2] = np.minimum(pel[:, 2], zreq)          # never exceed reach after smoothing

    # ---- IK
    q = np.zeros((n, len(JOINTS)))
    ext = np.zeros((n, 2))
    for k in range(n):
        for i, (s, sy) in enumerate((("L", 1), ("R", -1))):
            qi, ext[k, i] = leg_ik(pel[k], body_R[k], ank[s][k], footR[s][k],
                                   np.array([0, sy * geom.hip_y, -geom.hip_drop]),
                                   geom.thigh, geom.shank)
            q[k, 6 * i:6 * i + 6] = qi
        # arms: contralateral swing; elbow flexes more on the forward swing
        q[k, 12] = -p.arm_swing * (-d[k])      # left arm forward when right foot ahead
        q[k, 15] = -p.arm_swing * (d[k])
        q[k, 13] = p.arm_abduct
        q[k, 16] = -p.arm_abduct
        q[k, 14] = -(p.elbow_bias + 0.4 * max(0, -q[k, 12]))
        q[k, 17] = -(p.elbow_bias + 0.4 * max(0, -q[k, 15]))
        q[k, 18] = -0.5 * yaw[k]               # head counter-rotates: keeps gaze (and camera) forward

    return dict(t=t, q=q, joints=JOINTS, pelvis=pel, body_R=np.array(body_R), yaw=yaw,
                foot=foot, pitch=pitch, ankle=ank, zmp_ref=zmp_ref,
                zmp_model=np.c_[zmpx, zmpy], com=np.c_[comx[:, 0], comy[:, 0]],
                ext=ext, steps=steps, geom=geom, params=p)


def export_csv(traj, path):
    hdr = "t," + ",".join(traj["joints"])
    np.savetxt(path, np.c_[traj["t"], traj["q"]], delimiter=",", header=hdr, comments="", fmt="%.6f")


if __name__ == "__main__":
    tr = generate()
    export_csv(tr, "gait_joint_trajectory.csv")
    print("samples:", len(tr["t"]), " duration: %.2fs" % tr["t"][-1])
