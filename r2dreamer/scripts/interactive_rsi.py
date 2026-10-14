"""Interactive RSI ("Director Mode") -- v1, offline pose authoring.

Opens the robot in an interactive MuJoCo viewer. Drag it into a pose with
the mouse (ctrl + right-click-drag applies a force toward the cursor --
standard MuJoCo perturbation, nothing custom). While dragging, a live
geometry readout prints to the terminal: height_error and tilt relative to
the standing_start keyframe, and the torso's lateral offset from center --
the same pure-geometry quantities standing_reward_terms and
walking_transfer_v2_reward_terms are built on (height_error in particular
is exactly what the 2026-10-07 height-sag reward bug turned on).

NOT included in this v1 (see r2dreamer/docs/INTERACTIVE_RSI_DESIGN_20261007.md):
  - The foot-pressure-sensor-dependent reward terms (contact_quality,
    load_balance_quality). Those read a bespoke per-foot FSR sensor array
    tightly coupled to the derived training scene's contact solver -- not
    safe to hand-reimplement here without risking silently-wrong numbers.
    Only pure forward-kinematics quantities (position/orientation) are
    shown, which are exactly reproducible regardless of contact backend.
  - IK-driven foot dragging (grab a foot, whole leg solves, other foot
    anchored). This v1 uses MuJoCo's native body-perturbation only, which
    is already hierarchy-respecting for FK purposes (dragging the head
    only moves the head/neck) but not a real IK solver yet.
  - IK-driven foot dragging is still not built: v1's live-inject mode below
    reuses the same MuJoCo native body-perturbation (FK-honest) authoring.

Live injection into a running training process (added 2026-10-08, see
r2dreamer/docs/INTERACTIVE_RSI_USAGE_20261007.md "Live injection" section):
pass --inject-rundir pointing at a LIVE run's logdir. Pressing 'I' then
writes injection_edited_state.npz into that directory; the training
process (polling for it once per outer loop iteration, see trainer.py's
_check_pose_injection) pauses, backs itself up, applies your pose to every
world (exact on world 0, small joint-angle noise on the rest), and resumes.
This requires the run directory to be reachable on the SAME filesystem the
training process reads from -- run this tool directly on the training
machine (or against a mounted/synced copy of that exact logdir), not from
an unrelated machine.

Usage:
    python scripts/interactive_rsi.py
    python scripts/interactive_rsi.py --start-pose 3   # start from pose "3"
    python scripts/interactive_rsi.py --inject-rundir /path/to/live/run/logdir

Controls:
    ctrl + right-click-drag   pull a part; its whole limb up to the hip/shoulder follows
                              (any direction, incl. sideways) and stays where dropped
    s                         commit the current qpos as a new scratch pose
    p                         pause the live run and load its world 0 here
    f                         pin the RIGHT foot to the floor (again: LEFT foot, again: off = torso lock).
                              With a foot pinned, bending that leg moves the BODY over the planted
                              foot (like the real robot) instead of dragging the foot backwards
    l                         lock/unlock the torso in the air (locked by default:
                              the robot cannot balance without the policy)
    up / down arrow           raise / lower the locked torso by 1 cm
    k                         reset to the standing keyframe (upright)
    double-click a part, then
    , / .                     turn that part's joint back / forward by 0.05 rad,
                              only around its own axis (no drag box)
    g                         add/remove the double-clicked part's joint to a GROUP: , and . then
                              turn every grouped joint at once, and ctrl+drag keeps them loose too
    n                         flip the double-clicked grouped joint's direction (mirror moves)
    x                         clear the group
    b                         balance: bend the leg(s) on the floor so the weight sits over the
                              support foot / feet (feet stay put) -- do this before 'I'
                              so the injected robot does not tip toward the lifted leg
    u                         undo the last B
                              (only active when --inject-rundir was passed)
    i                         write injection_edited_state.npz for a live run
                              (only active when --inject-rundir was passed)
    q / close window          quit
"""
import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent / "envs" / "mjx_walking" / "assets"
SCENE_XML = ASSETS_DIR / "humanoid_rubber_2mm_servo_v2_kp40.xml"  # match training servo model; see --scene
MANUAL_POSES_JSON = ASSETS_DIR / "manual_rsi_poses.json"
SCRATCH_POSES_JSON = ASSETS_DIR / "manual_rsi_poses_scratch.json"
STANDING_KEYFRAME = "standing_start"


def load_model():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    if (model.nq, model.nv, model.nu) != (26, 25, 19):
        raise ValueError(
            f"Unexpected robot contract: {(model.nq, model.nv, model.nu)} "
            "-- this scene no longer matches the training contract.")
    return model


def standing_height(model):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, STANDING_KEYFRAME)
    if key_id < 0:
        raise ValueError(f"Keyframe '{STANDING_KEYFRAME}' not found in scene")
    return float(model.key_qpos[key_id, 2])


def load_pose_qpos(model, pose_id: str):
    with open(MANUAL_POSES_JSON) as f:
        poses = json.load(f)["poses"]
    if pose_id not in poses:
        raise ValueError(f"Pose '{pose_id}' not found in {MANUAL_POSES_JSON}")
    return np.array(poses[pose_id]["qpos"], dtype=np.float64)


def geometry_readout(model, data, standing_h):
    """Pure forward-kinematics quantities -- exactly the formulas in
    env.py's docstrings (height_error, tilt, com_offset), not a guess.
    These do not depend on the contact solver, so they are exactly
    reproducible regardless of backend (CPU MuJoCo here vs the Warp-GPU
    backend used in real training).
    """
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    height_error = data.xpos[base_id, 2] - standing_h
    rot = data.xmat[base_id].reshape(3, 3)
    tilt = np.arccos(np.clip(rot[2, 2], -1., 1.))
    com_xy = data.subtree_com[0, :2]
    return dict(height_error=float(height_error), tilt_deg=float(np.degrees(tilt)),
                com_xy=com_xy.copy(), root_height=float(data.xpos[base_id, 2]))


# Footstep targets, exactly as training sets them after a reset/injection (env.py footstep_target_terms,
# first_behind_toe rule). Walking direction is world -y, sideways is world +x.
#   FIRST step (right foot): its HEEL must land no more than FIRST_BEHIND_TOE behind the LEFT foot's toe,
#   within DOT_RADIUS of the right foot's standing line sideways.  Drawn as an orange line across the right
#   foot's lane; anywhere past it counts.
#   Then each foot's heel lands DOT_STEP ahead of where it last stood (blue disc = left foot's next dot).
FIRST_BEHIND_TOE, DOT_STEP, DOT_RADIUS = .05, .03, .025
FORWARD, LATERAL = np.array([0., -1.]), np.array([1., 0.])


def pad_points(model, data, side):
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_rubber_pad_2mm_candidate")
    a = model.mesh_vertadr[model.geom_dataid[g]]
    v = model.mesh_vert[a:a + model.mesh_vertnum[model.geom_dataid[g]]]
    return g, v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]


def standing_dot_offsets(model):
    """Per foot (left, right): standing sideways offset of the pad centre from the root."""
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, STANDING_KEYFRAME))
    mujoco.mj_forward(model, d)
    return np.array([(d.geom_xpos[pad_points(model, d, side)[0], :2] - d.qpos[:2]) @ LATERAL
                     for side in ("left", "right")])


def heel_to_dots(model, data, lanes):
    """Per foot (left, right): (target xy, heel xy, distance, forward gap, sideways gap, lowest point z, ok).
    Right = first step (forward gap = line minus heel; <= 0 is past the line). Left = its next dot."""
    feet = []
    for k, side in enumerate(("left", "right")):
        g, pts = pad_points(model, data, side)
        f = pts[:, :2] @ FORWARD
        feet.append((f.min(), f.max(), data.geom_xpos[g, :2] @ LATERAL, pts[:, 2].min()))
    to_xy = lambda f, l: f * FORWARD + l * LATERAL
    rows = []
    for k in (0, 1):
        heel, toe, lat, low = feet[k]
        lane = data.qpos[:2] @ LATERAL + lanes[k]
        target = feet[0][1] - FIRST_BEHIND_TOE if k == 1 else heel + DOT_STEP
        fwd, side = target - heel, lane - lat
        dist = float(np.hypot(max(fwd, 0.) if k == 1 else fwd, side))
        ok = (fwd <= 0. and abs(side) < DOT_RADIUS) if k == 1 else dist < DOT_RADIUS
        rows.append((to_xy(target, lane), to_xy(heel, lat), dist, float(fwd), float(side), float(low), bool(ok)))
    return rows


def draw_dots(scn, rows):
    scn.ngeom = 0
    for k, (target, heel, dist, _, _, _, ok) in enumerate(rows):
        rgba = (np.array([.1, .9, .1, .9]) if ok else
                np.array([1., .5, 0., .9]) if k == 1 else np.array([.2, .4, 1., .5]))
        shapes = [(np.r_[heel, .004], [.005, 0., 0.], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([1., 1., 0., 1.]))]
        if k == 1:   # first-step line across the right foot's lane (heel must land past it)
            shapes.append((np.r_[target, .0008], [DOT_RADIUS, .003, .0008], mujoco.mjtGeom.mjGEOM_BOX, rgba))
        else:
            shapes.append((np.r_[target, .0008], [DOT_RADIUS, .0008, 0.], mujoco.mjtGeom.mjGEOM_CYLINDER, rgba))
        for pos, size, typ, col in shapes:
            if scn.ngeom >= scn.maxgeom:
                return
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], typ, np.asarray(size, float), pos, np.eye(3).ravel(),
                                col.astype(np.float32))
            scn.ngeom += 1


def balance_pose(model, data, act_qpos, iters=60):
    """'B': bend the leg(s) whose foot is on the floor so the body's centre of mass sits over that
    support (one foot: its centre; two feet: between them), keeping those feet exactly where they are.
    Moves the torso (position only, stays upright) and the supporting legs' joints; a lifted leg
    keeps its joint angles. Returns a short report."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    mujoco.mj_kinematics(model, data)
    pads = {side: pad_points(model, data, side)[1] for side in ("left", "right")}
    floor = min(p[:, 2].min() for p in pads.values())
    support = [side for side, pts in pads.items() if pts[:, 2].min() < floor + .003]
    keep = {}
    for side in support:
        bottom = pads[side][:len(pads[side]) // 2]           # pad mesh: bottom face first
        keep[side] = bottom[np.linspace(0, len(bottom) - 1, 4).astype(int)].copy()
    goal = np.mean(np.concatenate([pads[side][:, :2] for side in support]), axis=0)
    legs = [i for i, n in enumerate(names) if any(n.startswith(f"act_{side}_") for side in support)
            and any(k in n for k in ("hip", "thigh", "knee", "foot"))]
    idx = np.r_[[0, 1, 2], act_qpos[legs]].astype(int)
    lo = np.r_[[-np.inf] * 3, model.actuator_ctrlrange[legs, 0]]
    hi = np.r_[[np.inf] * 3, model.actuator_ctrlrange[legs, 1]]

    def residual(x):
        data.qpos[idx] = x
        mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data)
        res = [pad_points(model, data, side)[1][:len(pads[side]) // 2][
            np.linspace(0, len(pads[side]) // 2 - 1, 4).astype(int)].ravel() - keep[side].ravel() for side in support]
        return np.r_[np.concatenate(res) * 3., data.subtree_com[0, :2] - goal]

    x = data.qpos[idx].copy(); start = x.copy()
    for _ in range(iters):
        r0 = residual(x)
        J = np.stack([(residual(x + e) - r0) / 1e-4 for e in np.eye(len(x)) * 1e-4], 1)
        step = -np.linalg.solve(J.T @ J + 1e-4 * np.eye(len(x)), J.T @ r0)
        x = np.clip(x + np.clip(step, -.05, .05), lo, hi)
    r = residual(x)
    mujoco.mj_forward(model, data)
    off = np.linalg.norm(data.subtree_com[0, :2] - goal)
    moved = ", ".join(f"{names[i][4:]} {x[3 + k] - start[3 + k]:+.2f}" for k, i in enumerate(legs)
                      if abs(x[3 + k] - start[3 + k]) > .01)
    return (f"balanced over {' + '.join(support)} foot: centre of mass {off * 100:.1f} cm from the support centre "
            f"(feet held to {np.abs(r[:-2]).max() / 3 * 1000:.1f} mm). Changed: {moved or 'nothing'}")


def pin_root_to_foot(model, data, pad_geom, anchor):
    """Move the free-floating torso (rigidly, whole robot) so foot pad `pad_geom` is back at
    `anchor` = (position, rotation matrix): the foot stays planted and the body moves instead."""
    pa, Ra = anchor
    pc, Rc = data.geom_xpos[pad_geom].copy(), data.geom_xmat[pad_geom].reshape(3, 3).copy()
    dR = Ra @ Rc.T
    root_rot = np.zeros(9)
    mujoco.mju_quat2Mat(root_rot, data.qpos[3:7])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (dR @ root_rot.reshape(3, 3)).ravel())
    data.qpos[:3] = pa + dR @ (data.qpos[:3] - pc)
    data.qpos[3:7] = quat
    data.qvel[:6] = 0.
    mujoco.mj_forward(model, data)


def next_scratch_pose_id():
    if not SCRATCH_POSES_JSON.exists():
        return "1"
    with open(SCRATCH_POSES_JSON) as f:
        existing = json.load(f).get("poses", {})
    existing_ids = [int(k) for k in existing if k.isdigit()]
    return str(max(existing_ids, default=0) + 1)


def commit_scratch_pose(model, data):
    if SCRATCH_POSES_JSON.exists():
        with open(SCRATCH_POSES_JSON) as f:
            bundle = json.load(f)
    else:
        bundle = {"source_model": str(SCENE_XML), "poses": {}}
    pose_id = next_scratch_pose_id()
    bundle["poses"][pose_id] = dict(
        qpos=data.qpos.tolist(),
        ctrl=data.ctrl.tolist(),
        committed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(SCRATCH_POSES_JSON, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"\n*** Committed scratch pose '{pose_id}' to {SCRATCH_POSES_JSON.name} ***")
    print(f"To use it in training, launch/resume with:")
    print(f"  pose_rsi_json=envs/mjx_walking/assets/{SCRATCH_POSES_JSON.name} "
          f"training_pose_id={pose_id} rsi_probabilities=[0.0,0.0,0.0,1.0]\n")
    return pose_id


def write_injection_request(rundir: Path, data, noise_std: float, seed):
    """Write injection_edited_state.npz into a live run's logdir.

    trainer.py's _check_pose_injection polls for exactly this filename.
    Written to a .tmp path and renamed so the trainer never observes a
    partially-written file.
    """
    rundir = Path(rundir)
    if not rundir.is_dir():
        raise ValueError(f"--inject-rundir does not exist or is not a directory: {rundir}")
    final_path = rundir / "injection_edited_state.npz"
    tmp_path = rundir / "injection_edited_state.npz.tmp"
    kwargs = dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(), ctrl=data.ctrl.copy(),
                  noise_std=np.asarray(noise_std, dtype=np.float64))
    if seed is not None:
        kwargs["seed"] = np.asarray(int(seed), dtype=np.int64)
    with open(tmp_path, "wb") as handle:
        np.savez(handle, **kwargs)
    tmp_path.rename(final_path)
    print(f"\n*** Wrote {final_path} ***")
    print("If training is paused waiting for a request, it will pick this up "
          "within ~1s and resume automatically.")
    print("(If training was not paused, it will pause at its next loop boundary and apply it.)\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-pose", type=str, default=None,
                        help="Manual RSI pose id (1-4) to start from; default is the standing keyframe")
    parser.add_argument("--inject-rundir", type=str, default=None,
                        help="Live run logdir to write injection_edited_state.npz into "
                             "(enables the 'I' key). Must be reachable on this filesystem.")
    parser.add_argument("--noise-std", type=float, default=0.03,
                        help="Std dev (rad) of per-world joint-angle noise applied to the "
                             "non-world-0 workers on injection (default 0.03)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional seed for the injection noise, for reproducibility")
    parser.add_argument("--scene", type=str, default=None,
                        help="Scene XML in envs/mjx_walking/assets (default: humanoid_rubber_2mm_servo.xml)")
    args = parser.parse_args()
    if args.scene:
        global SCENE_XML
        SCENE_XML = ASSETS_DIR / args.scene
    inject_rundir = Path(args.inject_rundir) if args.inject_rundir else None

    model = load_model()
    data = mujoco.MjData(model)
    standing_h = standing_height(model)

    if args.start_pose is not None:
        data.qpos[:] = load_pose_qpos(model, args.start_pose)
    else:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, STANDING_KEYFRAME)
        mujoco.mj_resetDataKeyframe(model, data, key_id)  # qpos and ctrl
    mujoco.mj_forward(model, data)

    print(__doc__)
    print(f"Standing-height reference: {standing_h:.4f} m")
    print("Viewer open. ctrl+right-click-drag to pose. Press 'S' (in the viewer "
          "window, not this terminal) to commit the current pose as a scratch RSI pose.")
    if inject_rundir is not None:
        print(f"Live injection ENABLED -> {inject_rundir}. Press 'P' to pause the run "
              "and load its world 0, pose it, then 'I' to inject and resume.")
    else:
        print("Live injection disabled (no --inject-rundir passed).")

    commit_requested = [False]
    keyframe_requested = [False]
    turn_requested = [0.]
    group = {}                 # actuator index -> direction (+1/-1) for multi-joint turns ('G', 'N', 'X')
    group_key = [None]
    balance_requested = [False]
    undo_requested = [False]
    before_balance = [None]
    inject_requested = [False]
    pause_requested = [False]
    waiting_for_live = [False]
    root_locked = [True]
    act_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    act_body = model.jnt_bodyid[model.actuator_trnid[:, 0]]
    act_dof = model.jnt_dofadr[model.actuator_trnid[:, 0]]

    def dragged_joints(body):
        # Actuators of the grabbed body; a jointless part (e.g. a sole) uses its parent's.
        while body > 0 and not (act_body == body).any():
            body = model.body_parentid[body]
        return act_body == body

    base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    def limb_chain(body):
        # Every joint from the grabbed part up to the torso (hip / shoulder included), so a drag
        # can swing the whole leg in any direction -- e.g. sideways needs the hip ROLL servo,
        # which a thigh drag alone would leave holding the leg in the middle.
        mask = np.zeros(model.nu, dtype=bool)
        while body > 0 and body != base_body:
            mask |= act_body == body
            body = model.body_parentid[body]
        return mask
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, STANDING_KEYFRAME)

    pad_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_rubber_pad_2mm_candidate")
                 for side in ("left", "right")]
    foot_pin = [None]          # None, 0 = left foot pinned, 1 = right foot pinned ('F' cycles)
    pin_anchor = [None]
    pin_requested = [False]

    def hold_pose():
        # Motors hold the current joint angles; torso lock pins the free joint here,
        # foot pin keeps the pinned foot exactly where it is now.
        data.ctrl[:] = data.qpos[act_qpos]
        data.qvel[:] = 0.
        root_pose[:] = data.qpos[:7]
        if foot_pin[0] is not None:
            g = pad_geoms[foot_pin[0]]
            pin_anchor[0] = (data.geom_xpos[g].copy(), data.geom_xmat[g].reshape(3, 3).copy())

    data.qvel[:] = 0.
    root_pose = data.qpos[:7].copy()
    hold_pose()
    good_state = [data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()]
    print("Torso LOCKED in the air (press L to release, arrows to raise/lower, K to stand up).")

    def key_callback(keycode):
        if keycode == ord("S"):
            commit_requested[0] = True
        elif keycode == ord("I") and inject_rundir is not None:
            inject_requested[0] = True
        elif keycode == ord("P") and inject_rundir is not None:
            pause_requested[0] = True
        elif keycode == ord("F"):
            pin_requested[0] = True
        elif keycode == ord("L"):
            foot_pin[0] = None
            root_locked[0] = not root_locked[0]
            root_pose[:] = data.qpos[:7]
            print(f"\nTorso {'LOCKED' if root_locked[0] else 'RELEASED (it will fall)'}")
        elif keycode in (265, 264):  # GLFW up / down arrow
            root_pose[2] += .01 if keycode == 265 else -.01
        elif keycode == ord("K"):
            keyframe_requested[0] = True
        elif keycode in (ord("G"), ord("N"), ord("X")):
            group_key[0] = chr(keycode)
        elif keycode == ord("B"):
            balance_requested[0] = True
        elif keycode == ord("U"):
            undo_requested[0] = True
        elif keycode in (ord(","), ord(".")):
            turn_requested[0] += .05 if keycode == ord(".") else -.05

    last_print = 0.0
    dot_offsets = standing_dot_offsets(model)
    print("EVERY pose you inject counts as the first step (+10 at once, credited to the foot furthest "
          "forward; +20 more if the robot stays up 1 s). The motors hold your pose 2 s, then ease into the gait. "
          f"After that each heel lands {DOT_STEP * 100:.0f} cm ahead of where it was, zig-zag (blue disc = left "
          "foot's next dot). The orange line / 'STEP OK' is only a guide for a natural step length.")
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            with viewer.lock():
                if keyframe_requested[0]:
                    keyframe_requested[0] = False
                    mujoco.mj_resetDataKeyframe(model, data, key_id)
                    hold_pose()
                if group_key[0]:
                    k, group_key[0] = group_key[0], None
                    own = np.flatnonzero(dragged_joints(viewer.perturb.select)) if viewer.perturb.select > 0 else []
                    if k == "X":
                        group.clear()
                    elif not len(own):
                        print("\nDouble-click a body part first, then G (add/remove) or N (flip its direction)")
                    elif k == "G":
                        a = int(own[0])
                        group.pop(a) if a in group else group.__setitem__(a, 1.)
                    elif k == "N" and int(own[0]) in group:
                        group[int(own[0])] *= -1.
                    print("\nGroup: " + (", ".join(
                        f"{'+' if d > 0 else '-'}{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)[4:]}"
                        for a, d in group.items()) or "empty (, and . turn the double-clicked part only)"))
                if pin_requested[0]:
                    pin_requested[0] = False
                    foot_pin[0] = {None: 1, 1: 0, 0: None}[foot_pin[0]]
                    root_locked[0] = foot_pin[0] is None
                    hold_pose()
                    print("\n" + ("Foot pin OFF: torso LOCKED in the air again" if foot_pin[0] is None else
                                  f"{('LEFT', 'RIGHT')[foot_pin[0]]} foot PINNED to the floor: bending its leg now moves "
                                  "the body over the planted foot (F again = other foot / off)"))
                if balance_requested[0]:
                    balance_requested[0] = False
                    before_balance[0] = (data.qpos.copy(), data.ctrl.copy())
                    print("\n" + balance_pose(model, data, act_qpos) + "   (U = undo)")
                    hold_pose()
                if undo_requested[0]:
                    undo_requested[0] = False
                    if before_balance[0] is None:
                        print("\nNothing to undo (U undoes the last B)")
                    else:
                        data.qpos[:], data.ctrl[:] = before_balance[0]
                        before_balance[0] = None
                        hold_pose()
                        mujoco.mj_forward(model, data)
                        print("\nUndid the balance: pose is back to how you left it")
                if turn_requested[0] and group:
                    turn, turn_requested[0] = turn_requested[0], 0.
                    for a, sign in group.items():
                        low, high = model.actuator_ctrlrange[a]
                        angle = np.clip(data.ctrl[a] + sign * turn, low, high)
                        data.ctrl[a] = angle
                        data.qpos[act_qpos[a]] = angle
                        data.qvel[act_dof[a]] = 0.
                    mujoco.mj_forward(model, data)
                    print("\n" + ", ".join(f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)[4:]} "
                                           f"{data.ctrl[a]:+.2f}" for a in group))
                if turn_requested[0]:
                    turn, turn_requested[0] = turn_requested[0], 0.
                    joint = dragged_joints(viewer.perturb.select)
                    if viewer.perturb.select > 0 and joint.any():
                        low, high = model.actuator_ctrlrange[joint][0]
                        angle = np.clip(data.ctrl[joint][0] + turn, low, high)
                        data.ctrl[joint] = angle
                        data.qpos[act_qpos[joint]] = angle
                        data.qvel[act_dof[joint]] = 0.
                        mujoco.mj_forward(model, data)
                        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                                 int(np.flatnonzero(joint)[0]))
                        print(f"\n{name}: {angle:+.2f} rad (limits {low:+.2f}..{high:+.2f})")
                    else:
                        print("\nDouble-click a body part first, then use , and .")
                if viewer.perturb.active:
                    # The grabbed limb gives way from that part up to the torso (and stays where
                    # dropped); every other motor keeps holding its pose.
                    limp = limb_chain(viewer.perturb.select)
                    limp[list(group)] = True       # grouped joints give way too
                    data.ctrl[limp] = data.qpos[act_qpos[limp]]
                mujoco.mj_step(model, data)
                if viewer.perturb.active:
                    # Pin every other joint at its held angle: the motors alone are too weak.
                    data.qpos[act_qpos[~limp]] = data.ctrl[~limp]
                    data.qvel[act_dof[~limp]] = 0.
                    mujoco.mj_forward(model, data)
                if foot_pin[0] is not None:
                    pin_root_to_foot(model, data, pad_geoms[foot_pin[0]], pin_anchor[0])
                elif root_locked[0]:
                    data.qpos[:7] = root_pose
                    data.qvel[:6] = 0.
                    mujoco.mj_forward(model, data)
                if np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all():
                    good_state[:] = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
                else:
                    # A hard drag can blow up the physics (renders as a black window).
                    mujoco.mj_resetData(model, data)
                    data.qpos[:], data.qvel[:], data.ctrl[:] = good_state[0], good_state[1], good_state[2]
                    hold_pose()
                    mujoco.mj_forward(model, data)
                    print("\n*** Physics went unstable (drag too hard?); restored last good pose ***")
            dots = heel_to_dots(model, data, dot_offsets)
            draw_dots(viewer.user_scn, dots)
            viewer.sync()
            if commit_requested[0]:
                data.ctrl[:] = data.qpos[act_qpos]
                commit_scratch_pose(model, data)
                commit_requested[0] = False
            if pause_requested[0]:
                pause_requested[0] = False
                flag = inject_rundir / "pose_injection_request.flag"
                if not flag.exists():  # else a pause is already pending/active: keep its snapshot
                    (inject_rundir / "injection_current_state.npz").unlink(missing_ok=True)
                    # A leftover pose from an earlier 'I' would be applied the instant training
                    # pauses, before you can edit -- drop it so P always waits for a fresh 'I'.
                    (inject_rundir / "injection_edited_state.npz").unlink(missing_ok=True)
                    flag.touch()
                waiting_for_live[0] = True
                print("\n*** Pause requested; waiting for training to stop at its next loop "
                      "boundary (can take a while during optimizer updates)... ***")
            if waiting_for_live[0] and (inject_rundir / "injection_current_state.npz").exists():
                waiting_for_live[0] = False
                with viewer.lock():
                    live = np.load(inject_rundir / "injection_current_state.npz")
                    data.qpos[:], data.qvel[:], data.ctrl[:] = live["qpos"], live["qvel"], live["ctrl"]
                    hold_pose()
                    mujoco.mj_forward(model, data)
                print(f"\n*** Training paused. Loaded live world 0 (episode step "
                      f"{int(live['step'])}). Pose it, then press 'I' to inject + resume. ***")
            if inject_requested[0]:
                data.ctrl[:] = data.qpos[act_qpos]  # injected motors hold the authored pose
                data.qvel[:] = 0.
                write_injection_request(inject_rundir, data, args.noise_std, args.seed)
                # Without a pending pause the file would sit unread; request one so training picks
                # the pose up at its next loop boundary either way.
                (inject_rundir / "pose_injection_request.flag").touch()
                inject_requested[0] = False
            now = time.time()
            if now - last_print > 0.5:
                g = geometry_readout(model, data, standing_h)
                if inject_rundir is not None:
                    try:  # live pose for remote checking ("check my pose"): qpos/ctrl + heel-to-dot numbers
                        tmp = inject_rundir / "editor_view.npz.tmp"
                        with open(tmp, "wb") as handle:
                            np.savez(handle, qpos=data.qpos, ctrl=data.ctrl, time=np.float64(now),
                                     dot_dist=np.array([r[2] for r in dots]), dot_fwd=np.array([r[3] for r in dots]),
                                     dot_side=np.array([r[4] for r in dots]), foot_low=np.array([r[5] for r in dots]),
                                     step_ok=np.array([r[6] for r in dots]))
                        tmp.rename(inject_rundir / "editor_view.npz")
                    except OSError:
                        pass
                print(f"\rR heel vs (L toe - {FIRST_BEHIND_TOE * 100:.0f}cm): {-dots[1][3] * 100:+5.1f}cm "
                      f"side {dots[1][4] * 100:+.1f}cm{' STEP OK' if dots[1][6] else ''} "
                      f"lift {dots[1][5] * 1000:+.0f}mm | height_error={g['height_error']:+.4f}m  "
                      f"tilt={g['tilt_deg']:5.1f}deg  "
                      f"root_height={g['root_height']:.4f}m  "
                      f"com_xy=({g['com_xy'][0]:+.3f},{g['com_xy'][1]:+.3f})   ",
                      end="", flush=True)
                last_print = now
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
