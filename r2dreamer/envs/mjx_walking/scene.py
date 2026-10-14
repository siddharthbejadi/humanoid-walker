"""Build a sensor-augmented *copy* of the 2 mm sole scene for MJX walking.

The production and source candidate XML files are never edited.  MJX-Warp's
public Data API exposes contact sensors, not its packed contact buffers.  Two
fixed-size contact sensors therefore expose normal forces and positions for
the same foot/ground pairs used by the eight-channel FSR calculation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


SENSOR_NAMES = ("mjx_left_sole_contacts", "mjx_right_sole_contacts")
CONTACTS_PER_FOOT = 32
CONTACT_FIELDS = 7  # found, force[3], world position[3]
VENDOR_GAIT_CHANNELS = (12, 16, 15, 14, 13, 23, 19, 20, 21, 22,
                        3, 2, 1, 8, 9, 10, 4, 0, 11)


def build_mjx_scene(source: Path, *, motor_torque_kgcm: float | None = None) -> Path:
    """Return an immutable, adjacent MJCF copy with two contact sensors.

    An adjacent output preserves relative mesh/include references.  Existing
    output is accepted only if byte-identical to what this function generates.
    """
    source = Path(source).resolve(strict=True)
    tree = ET.parse(source)
    root = tree.getroot()
    if root.tag != "mujoco":
        raise ValueError("Expected a MuJoCo MJCF root")
    if motor_torque_kgcm is not None:
        import math
        if not math.isfinite(motor_torque_kgcm) or motor_torque_kgcm <= 0:
            raise ValueError("Motor torque must be a positive finite kg-cm value")
        torque_nm = motor_torque_kgcm * 0.0980665
        limit = f"{-torque_nm:.6f} {torque_nm:.6f}"
        actuators = root.findall("actuator/position")
        if len(actuators) != 19:
            raise ValueError("Motor torque override requires 19 position actuators")
        joints = {joint.get("name"): joint for joint in root.iter("joint")}
        for actuator in actuators:
            joint = joints.get(actuator.get("joint"))
            if joint is None:
                raise ValueError(f"Missing actuator joint: {actuator.get('joint')}")
            actuator.set("forcerange", limit)
            joint.set("actuatorfrcrange", limit)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF lacks worldbody")
    grounds = [geom for geom in worldbody.findall("geom") if geom.get("type") == "plane"]
    if len(grounds) != 1:
        raise ValueError(f"Expected one world-ground plane, found {len(grounds)}")
    ground = grounds[0]
    if not ground.get("name"):
        ground.set("name", "mjx_walk_ground")
    geom_names = {geom.get("name") for geom in root.iter("geom")}
    pads = []
    for side in ("left", "right"):
        candidates = [name for name in geom_names if name and name.startswith(side + "_foot_rubber")]
        if len(candidates) != 1:
            raise ValueError(f"Expected one named {side} 2 mm rubber sole geom, found {candidates}")
        pads.append(candidates[0])
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    existing = {item.get("name") for item in sensor}
    if any(name in existing for name in SENSOR_NAMES):
        raise ValueError("Source scene already defines MJX sole sensors")
    for pad, name in zip(pads, SENSOR_NAMES):
        ET.SubElement(sensor, "contact", {
            "name": name, "geom1": pad, "geom2": ground.get("name"),
            "num": str(CONTACTS_PER_FOOT), "reduce": "maxforce",
            "data": "found force pos",
        })
    suffix = ("_mjx_contacts" if motor_torque_kgcm is None
              else f"_mjx_contacts_{motor_torque_kgcm:g}kgcm")
    target = source.with_name(source.stem + suffix + ".xml")
    content = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    if target.exists():
        if target.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite differing MJX scene: {target}")
    else:
        target.write_bytes(content)
    return target


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rsi_injection_flags(pose_ids, *, manual_pool: bool, injected_pose_min_id: int):
    """Only authored manual poses may activate hold/blend and injection credit."""
    import numpy as np

    ids = np.asarray(pose_ids)
    return (ids >= injected_pose_min_id) & bool(manual_pool)


def build_scripted_rsi_pool(model, path: Path):
    """Load finite phase states without first constructing an unrelated pose pool."""
    import numpy as np

    source = Path(path).resolve(strict=True)
    with np.load(source, allow_pickle=False) as archive:
        required = {"qpos", "qvel", "ctrl", "phase_offset", "n"}
        if not required.issubset(archive.files):
            raise ValueError("Scripted RSI pool is missing required state arrays")
        size = archive["n"]
        if size.shape != () or size.dtype.kind not in "iu" or int(size) < 1:
            raise ValueError("Scripted RSI pool needs a positive integer phase count")
        n = int(size)
        arrays = {name: archive[name].copy() for name in required - {"n"}}
    shapes = {"qpos": (n + 1, model.nq), "qvel": (n + 1, model.nv),
              "ctrl": (n + 1, model.nu), "phase_offset": (n + 1,)}
    for name, shape in shapes.items():
        value = arrays[name]
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid scripted RSI {name}: expected finite {shape}")
    phases = arrays["phase_offset"]
    if phases.dtype.kind not in "iu" or np.any(phases < 0):
        raise ValueError("Scripted RSI phases must be nonnegative integers")
    ctrl = arrays["ctrl"]
    if (np.any(ctrl < model.actuator_ctrlrange[:, 0] - 1e-6)
            or np.any(ctrl > model.actuator_ctrlrange[:, 1] + 1e-6)):
        raise ValueError("Scripted RSI controls exceed the selected actuator limits")
    if not np.allclose(np.linalg.norm(arrays["qpos"][:, 3:7], axis=1), 1., atol=1e-3):
        raise ValueError("Scripted RSI root quaternions are not normalized")
    return dict(**arrays, pose_ids=np.r_[0, np.arange(n) + 100].astype(np.int32),
                counts=np.asarray([1, 0, 0, n], np.int32),
                offsets=np.asarray([0, 1, 1, 1], np.int32),
                stats=dict(source="scripted phase pool", states=n))


def balanced_pose_reset_indices(pose_ids, worlds: int):
    """Map an equal, stable pose-ID schedule to the pool's storage order.

    This is evaluation-only: training keeps its configured categorical draw.
    Manual poses are exact starts, so repeats of a pose are repeats, not new
    independent randomized starts.
    """
    import numpy as np

    ids = np.asarray(pose_ids)
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu"
            or np.any(ids < 0) or 0 not in ids or len(np.unique(ids)) != len(ids)):
        raise ValueError("Balanced evaluation requires unique nonnegative pose IDs including standing ID 0")
    if worlds < 1 or worlds % len(ids):
        raise ValueError(
            f"Balanced evaluation needs a positive world count divisible by {len(ids)} pose IDs; got {worlds}")
    return np.tile(np.argsort(ids), worlds // len(ids)).astype(np.int32)


def grouped_pose_reset_indices(pose_ids, worlds: int, groups):
    """Fixed pose per world: worlds split into len(groups) equal contiguous blocks.

    groups lists pose IDs (0 = standing). 1024 worlds over [0,1,2,3,4] gives
    205/205/205/205/204. Every reset of a world returns it to its own pose.
    """
    import numpy as np

    ids = np.asarray(pose_ids)
    groups = [int(g) for g in groups]
    if worlds < len(groups) or not groups:
        raise ValueError(f"Pose groups need at least {len(groups)} worlds; got {worlds}")
    index_of = {}
    for g in groups:
        hit = np.flatnonzero(ids == g)
        if len(hit) != 1:
            raise ValueError(f"Pose group {g} is absent (or duplicated) in the reset pool {sorted(ids.tolist())}")
        index_of[g] = int(hit[0])
    block = np.arange(worlds) * len(groups) // worlds
    return np.asarray([index_of[groups[b]] for b in block], dtype=np.int32)


def training_pose_reset_index(pose_ids, pose_id):
    """Resolve one exact manual-pose ID without changing the evaluation pool."""
    import numbers
    import numpy as np

    if isinstance(pose_id, (bool, np.bool_)) or not isinstance(pose_id, numbers.Integral):
        raise ValueError("training_pose_id must be an integer in [0, 4]")
    if not 0 <= pose_id <= 4:
        raise ValueError("training_pose_id must be an integer in [0, 4]")
    ids = np.asarray(pose_ids)
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("Training pose selection requires unique integer pool pose IDs")
    selected = np.flatnonzero(ids == pose_id)
    if len(selected) != 1:
        raise ValueError(f"Training pose ID {pose_id} is absent from the reset pool")
    return int(selected[0])


def build_pose_rsi_pool(model, json_path: Path, action_scale,
                        *, min_clearance: float = .020, hold_seconds: float = 2.,
                        allow_unstable: bool = False):
    """Build manual reset states, with optional unqualified exploration starts.

    The strict path requires reachable, collision-free poses that survive a
    held-control CPU replay. The explicit exploratory path projects commands
    into the actor range and admits falling and partial-clearance poses.
    It preserves the original JSON unchanged and is not a balance certificate.
    """
    import mujoco
    import numpy as np

    source = Path(json_path).expanduser().resolve(strict=True)
    entries = json.loads(source.read_text(encoding="utf-8")).get("poses")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Pose RSI JSON needs a nonempty 'poses' mapping")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "standing_start")
    if key_id < 0 or model.nu != 19:
        raise ValueError("Pose RSI requires standing_start and 19 actuators")
    scale = np.asarray(action_scale, dtype=np.float64)
    if scale.shape != (19,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Invalid 19-channel actor action scale")
    qpos_adrs = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    nominal = np.asarray(model.key_qpos[key_id], dtype=np.float64)[qpos_adrs]
    sole_ids = []
    sole_vertices = []
    for side in ("left", "right"):
        ids = [i for i in range(model.ngeom)
               if model.geom(i).name.startswith(side + "_foot_rubber")]
        if len(ids) != 1:
            raise ValueError(f"Pose RSI needs one {side} rubber sole geom")
        geom_id = ids[0]
        mesh_id = int(model.geom_dataid[geom_id])
        first, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        sole_ids.append(geom_id)
        sole_vertices.append(np.asarray(model.mesh_vert[first:first + count]).copy())
    grounds = {i for i in range(model.ngeom)
               if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE}
    if len(grounds) != 1:
        raise ValueError("Pose RSI needs exactly one ground plane")
    ground = next(iter(grounds))
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    standing = np.asarray(model.key_qpos[key_id], dtype=np.float64)
    final_category = "other_pose" if allow_unstable else "second_step"
    pools = {"standing": [(standing.copy(), np.zeros(model.nv), nominal.copy(), 0, 0)],
             "left_swing": [], "right_swing": [], final_category: []}
    stats = {"accepted": [], "rejected": {}, "projected_deg": {},
             "mode": "exploratory_unstable" if allow_unstable else "strict_stable"}
    data = mujoco.MjData(model)

    def sole_height(index):
        geom_id = sole_ids[index]
        points = (sole_vertices[index] @ data.geom_xmat[geom_id].reshape(3, 3).T
                  + data.geom_xpos[geom_id])
        return float(np.min(points[:, 2]))

    def has_invalid_contact():
        stance_contact = [False, False]
        for contact in data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            for side, sole_id in enumerate(sole_ids):
                if pair == {ground, sole_id}:
                    stance_contact[side] = True
            if contact.dist < -.0001 and pair != {ground, sole_ids[0]} and pair != {ground, sole_ids[1]}:
                if ground in pair or all(model.geom_bodyid[g] > 0 for g in pair):
                    return stance_contact, True
        return stance_contact, False

    for ordinal, (name, entry) in enumerate(entries.items(), start=1):
        try:
            pose_id = int(name) if str(name).isdigit() else ordinal
            pose = np.asarray(entry["qpos"], dtype=np.float64)
            ctrl = np.asarray(entry["ctrl"], dtype=np.float64)
            if (pose.shape != (model.nq,) or ctrl.shape != (19,)
                    or not np.all(np.isfinite(pose)) or not np.all(np.isfinite(ctrl))):
                raise ValueError("invalid qpos/ctrl shape or nonfinite value")
            if abs(np.linalg.norm(pose[3:7]) - 1.) > 1e-3:
                raise ValueError("root quaternion is not normalized")
            if np.max(np.abs(pose[qpos_adrs] - ctrl)) > 1e-4:
                raise ValueError("saved motor commands do not match joint positions")
            low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            if np.any(ctrl < low - 1e-6) or np.any(ctrl > high + 1e-6):
                raise ValueError("motor command exceeds a physical joint limit")
            lower = np.maximum(low, nominal - scale)
            upper = np.minimum(high, nominal + scale)
            if allow_unstable:
                original_ctrl = ctrl.copy()
                ctrl = np.clip(ctrl, lower, upper)
                pose = pose.copy()
                pose[qpos_adrs] = ctrl
                stats["projected_deg"][str(name)] = round(float(np.degrees(
                    np.max(np.abs(original_ctrl - ctrl)))), 3)
            else:
                excess = np.abs(ctrl - nominal) - scale
                if np.any(excess > 1e-6):
                    channel = int(np.argmax(excess))
                    raise ValueError(
                        f"{model.actuator(channel).name} needs "
                        f"{np.degrees(abs(ctrl[channel] - nominal[channel])):.1f} deg "
                        f"from standing; policy permits {np.degrees(scale[channel]):.1f} deg")
            mujoco.mj_resetData(model, data)
            data.qpos[:] = pose
            data.ctrl[:] = ctrl
            mujoco.mj_forward(model, data)
            heights = [sole_height(0), sole_height(1)]
            swing = int(np.argmax(heights))
            stance = 1 - swing
            if allow_unstable:
                category = (("left_swing", "right_swing")[swing]
                            if heights[swing] - heights[stance] >= min_clearance
                            else final_category)
                pools[category].append((pose.copy(), np.zeros(model.nv), ctrl.copy(), 0, pose_id))
                stats["accepted"].append(dict(name=str(name), category=category,
                    clearance_mm=round((heights[swing] - heights[stance]) * 1000, 2)))
                continue
            if heights[swing] - heights[stance] < min_clearance:
                raise ValueError(
                    f"swing sole clearance {(heights[swing] - heights[stance]) * 1000:.1f} mm"
                    f" < {min_clearance * 1000:g} mm")
            if not -.002 <= heights[stance] <= .003:
                raise ValueError(f"stance sole height {heights[stance] * 1000:.1f} mm is outside contact range")
            contacts, invalid = has_invalid_contact()
            if invalid or not contacts[stance]:
                raise ValueError("no sole-only stance support or invalid penetration")
            original_height = float(pose[2])
            steps = int(np.ceil(hold_seconds / model.opt.timestep))
            for _ in range(steps):
                mujoco.mj_step(model, data)
                tilt = float(np.arccos(np.clip(data.xmat[base_id].reshape(3, 3)[2, 2], -1., 1.)))
                contacts, invalid = has_invalid_contact()
                if (not np.all(np.isfinite(data.qpos)) or tilt >= .35
                        or data.qpos[2] < original_height - .03
                        or invalid or not any(contacts)):
                    raise ValueError(f"held-control replay lost support at {data.time:.3f} s")
            pools[("left_swing", "right_swing")[swing]].append(
                (pose.copy(), np.zeros(model.nv), ctrl.copy(), 0, pose_id))
            stats["accepted"].append(str(name))
        except (KeyError, TypeError, ValueError) as exc:
            stats["rejected"][str(name)] = str(exc)
    names = ("standing", "left_swing", "right_swing", final_category)
    counts = np.asarray([len(pools[name]) for name in names], dtype=np.int32)
    offsets = np.cumsum(np.r_[0, counts[:-1]]).astype(np.int32)
    records = [record for name in names for record in pools[name]]
    return dict(qpos=np.stack([r[0] for r in records]).astype(np.float32),
                qvel=np.stack([r[1] for r in records]).astype(np.float32),
                ctrl=np.stack([r[2] for r in records]).astype(np.float32),
                phase_offset=np.asarray([r[3] for r in records], dtype=np.int32),
                pose_ids=np.asarray([r[4] for r in records], dtype=np.int32),
                counts=counts, offsets=offsets, category_names=names, stats=stats)


def vendor_gait_reference(model, csv_path: Path, signs, dt: float, horizon_s: float,
                          *, cycle=(6, 10), speed: float = 1.):
    """Joint-target table for imitation: vendor start-up, then its stride cycle looped.

    vendor_forward_3_steps.csv: frames 1-6 shift weight and start, frames 7-10
    repeat exactly as 11-14 (one 3.0 s stride: left + right step), 15-16 stop.
    Returns (targets[steps, nu], cycle_start_s, cycle_period_s) on the control
    grid, smoothstep-interpolated between frames like the vendor servos, using
    the same pulse->radian conversion as build_vendor_rsi_pool.
    """
    import mujoco
    import numpy as np

    with Path(csv_path).expanduser().resolve(strict=True).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    durations = np.asarray([float(row["time_ms"]) / 1000. for row in rows]) / float(speed)
    pulses = np.asarray([[float(row[f"ch{c}"]) for c in VENDOR_GAIT_CHANNELS] for row in rows])
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "standing_start")
    nominal = model.key_qpos[key_id][model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    frames = np.clip(nominal + (pulses - pulses[0]) * (np.pi / 2000.) * np.asarray(signs, float), low, high)
    first, end = cycle
    keys, times, t = [nominal], [0.], 0.
    for i in range(first):  # start-up frames
        t += durations[i]; keys.append(frames[i]); times.append(t)
    cycle_start = t + durations[first]
    while t < horizon_s + 1.:
        for i in range(first, end):
            t += durations[i]; keys.append(frames[i]); times.append(t)
    period = float(np.sum(durations[first:end]))
    grid = np.arange(int(horizon_s / dt) + 1) * dt
    seg = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(times) - 2)
    u = np.clip((grid - np.asarray(times)[seg]) / np.diff(times)[seg], 0., 1.)
    blend = (u * u * (3. - 2. * u))[:, None]
    keys = np.asarray(keys)
    return (keys[seg] + blend * (keys[seg + 1] - keys[seg])).astype(np.float32), float(cycle_start), period


def scripted_gait_reference(npz_path, dt: float, horizon_s: float, *, speed: float = 1.):
    """Joint targets from scripts/scripted_walk.py (inverse-kinematics quasi-static walk that steps in
    this simulator, unlike the vendor pattern): a start-up part, then a repeating 2-step cycle.
    Same return as vendor_gait_reference: (targets[steps, nu], cycle_start_s, cycle_period_s)."""
    import numpy as np

    data = np.load(Path(npz_path).expanduser().resolve(strict=True))
    ctrl, start, period, src_dt = data["ctrl"], int(data["cycle_start"]), int(data["cycle_len"]), float(data["dt"])
    n_out = int(horizon_s / dt) + 1
    need = int(np.ceil(n_out * dt * speed / src_dt)) + 3
    idx = np.arange(need)
    base = ctrl[np.where(idx < start, idx, start + (idx - start) % period)]
    pos = np.arange(n_out) * dt * speed / src_dt
    lo = np.clip(np.floor(pos).astype(int), 0, need - 2)
    frac = (pos - lo)[:, None]
    table = base[lo] + frac * (base[lo + 1] - base[lo])
    return table.astype(np.float32), start * src_dt / speed, period * src_dt / speed


def build_vendor_rsi_pool(model, csv_path: Path, signs, action_scale,
                          *, speed: float = 1., min_clearance: float = .020):
    """Build reachable, collision-checked reset poses from the vendor commands.

    This is a kinematic reset source, not a demonstrated or dynamically stable
    gait. The CPU scan happens once at startup; only fixed arrays enter JAX.
    """
    import mujoco
    import numpy as np

    csv_path = Path(csv_path).expanduser().resolve(strict=True)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not np.isfinite(speed) or speed <= 0:
        raise ValueError("RSI reference requires frames and positive finite speed")
    durations = np.asarray([float(row["time_ms"]) / 1000. for row in rows])
    pulses = np.asarray([[float(row[f"ch{channel}"]) for channel in VENDOR_GAIT_CHANNELS]
                         for row in rows], dtype=np.float64)
    signs = np.asarray(signs, dtype=np.float64)
    action_scale = np.asarray(action_scale, dtype=np.float64)
    if (pulses.shape != (len(rows), 19) or signs.shape != (19,)
            or action_scale.shape != (19,) or not np.all(np.isin(signs, (-1., 1.)))
            or not np.all(np.isfinite(pulses)) or not np.all(np.isfinite(durations))
            or np.any(durations <= 0) or np.any(action_scale <= 0)):
        raise ValueError("Invalid 19-channel vendor RSI reference or action range")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "standing_start")
    if key_id < 0 or model.nu != 19:
        raise ValueError("Vendor RSI requires the standing_start 19-actuator model")
    standing = np.asarray(model.key_qpos[key_id], dtype=np.float64)
    qpos_adrs = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    dof_adrs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    nominal = standing[qpos_adrs]
    low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    targets = np.clip(nominal + (pulses - pulses[0]) * (np.pi / 2000.) * signs,
                      low, high)
    mirrored = targets.copy()
    for left, right in zip(range(5), range(5, 10)):
        mirrored[:, right] = nominal[right] + (targets[:, left] - nominal[left]) * signs[right] / signs[left]
        mirrored[:, left] = nominal[left] + (targets[:, right] - nominal[right]) * signs[left] / signs[right]
    mirrored = np.clip(mirrored, low, high)
    pad_ids = []
    pad_vertices = []
    for side in ("left", "right"):
        candidates = [i for i in range(model.ngeom)
                      if model.geom(i).name.startswith(side + "_foot_rubber")]
        if len(candidates) != 1:
            raise ValueError(f"RSI needs one {side} rubber sole geom")
        geom = candidates[0]
        mesh = int(model.geom_dataid[geom])
        begin, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        pad_ids.append(geom)
        pad_vertices.append(np.asarray(model.mesh_vert[begin:begin + count]).copy())
    ground_ids = {i for i in range(model.ngeom)
                  if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE}
    if len(ground_ids) != 1:
        raise ValueError("RSI needs exactly one ground plane")
    ground = next(iter(ground_ids))
    data = mujoco.MjData(model)
    bounds = np.cumsum(durations)
    pools = {"standing": [(standing.copy(), np.zeros(model.nv), nominal.copy(), 0)],
             "left_swing": [], "right_swing": [], "second_step": []}
    stats = {"airborne": 0, "reachable": 0, "contact_checked": 0,
             "minimum_airborne_target_range_rad": float("inf")}

    def sole_points(geom, vertices):
        return vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]

    for trajectory in (targets, mirrored):
        for segment, duration in enumerate(durations):
            previous = nominal if segment == 0 else trajectory[segment - 1]
            destination = trajectory[segment]
            for u in np.linspace(.05, .95, 20):
                blend = u * u * (3. - 2. * u)
                pose = previous + blend * (destination - previous)
                velocity = (destination - previous) * 6. * u * (1. - u) * speed / duration
                mujoco.mj_resetData(model, data)
                data.qpos[:] = standing
                data.qpos[qpos_adrs] = pose
                data.ctrl[:] = pose
                mujoco.mj_forward(model, data)
                heights = [float(np.min(sole_points(g, v)[:, 2]))
                           for g, v in zip(pad_ids, pad_vertices)]
                stance = int(np.argmin(heights))
                swing = 1 - stance
                clearance = heights[swing] - heights[stance]
                if clearance <= min_clearance:
                    continue
                stats["airborne"] += 1
                required_range = float(np.max(np.abs(pose - nominal)))
                stats["minimum_airborne_target_range_rad"] = min(
                    stats["minimum_airborne_target_range_rad"], required_range)
                # A reset outside the actor's target set would immediately be
                # driven away from its hold command on the next control step.
                if np.any(np.abs(pose - nominal) > action_scale + 1e-6):
                    continue
                stats["reachable"] += 1
                # Give the stance rubber a small preload; avoid relying on a
                # zero-distance contact which the solver may classify as absent.
                data.qpos[2] += -.00025 - heights[stance]
                data.qvel[dof_adrs] = velocity
                mujoco.mj_forward(model, data)
                stance_points = sole_points(pad_ids[stance], pad_vertices[stance])
                point = stance_points[np.argmin(stance_points[:, 2])]
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros_like(jacp)
                mujoco.mj_jac(model, data, jacp, jacr, point,
                              int(model.geom_bodyid[pad_ids[stance]]))
                data.qvel[:3] = -(jacp @ data.qvel)
                mujoco.mj_forward(model, data)
                if float(np.min(sole_points(pad_ids[swing], pad_vertices[swing])[:, 2])) <= min_clearance:
                    continue
                stance_contact = False
                invalid_contact = False
                for contact in data.contact:
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if pair == {ground, pad_ids[stance]}:
                        stance_contact = True
                    elif contact.dist < -.0001:
                        if ground in pair or all(model.geom_bodyid[g] > 0 for g in pair):
                            invalid_contact = True
                if invalid_contact or not stance_contact:
                    continue
                stats["contact_checked"] += 1
                # The walking command's stance is left in steps 0..99 and
                # right in 125..224; align the reset with measured support.
                phase_offset = int((10 if stance == 0 else 135) + 80 * u)
                record = (data.qpos.copy(), data.qvel.copy(), pose.copy(), phase_offset)
                pools[("left_swing", "right_swing")[swing]].append(record)
                if u >= .75:
                    pools["second_step"].append(record)
    names = ("standing", "left_swing", "right_swing", "second_step")
    counts = np.asarray([len(pools[name]) for name in names], dtype=np.int32)
    offsets = np.cumsum(np.r_[0, counts[:-1]]).astype(np.int32)
    records = [record for name in names for record in pools[name]]
    return dict(qpos=np.stack([r[0] for r in records]).astype(np.float32),
                qvel=np.stack([r[1] for r in records]).astype(np.float32),
                ctrl=np.stack([r[2] for r in records]).astype(np.float32),
                phase_offset=np.asarray([r[3] for r in records], dtype=np.int32),
                counts=counts, offsets=offsets, stats=stats)
