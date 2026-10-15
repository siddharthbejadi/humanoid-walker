"""Humanoid in MuJoCo with the right arm driven kinematically (no dynamics): shared by the live demo and the offline renderer."""
import json
from pathlib import Path
import mujoco, numpy as np

ASSET = Path(__file__).resolve().parent.parent / "assets" / "humanoid"
# The sim model mixes left/right names between limbs; these four joints are all on the robot's anatomical RIGHT arm (x<0; robot faces -y).
ARM_JOINTS = ["right_shoulder_revolute", "left_arm_revolute", "right_elbow_revolute", "left_hand_tip_revolute"]


class Humanoid:
    def __init__(self, width=480, height=480, stand_frame=0, sign=(1, 1, 1, 1)):
        self.model = mujoco.MjModel.from_xml_path(str(ASSET / "humanoid.xml"))
        self.data = mujoco.MjData(self.model)
        self.q0 = np.array(json.loads((ASSET / "standing_qpos.json").read_text())["qpos"])      # committed as JSON: the repo ignores *.npz
        self.adr = [self.model.joint(n).qposadr[0] for n in ARM_JOINTS]
        self.sign = np.array(sign, float)
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.cam = mujoco.MjvCamera(); self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = (0, 0, 0.18); self.cam.distance = 0.85; self.cam.azimuth = 90; self.cam.elevation = -5

    def render(self, angles):
        """angles: [shoulder, arm, elbow, gripper] rad -> RGB image. Rest of the body stays in the standing pose."""
        self.data.qpos[:] = self.q0
        for a, ang, s in zip(self.adr, angles, self.sign):
            self.data.qpos[a] = s * ang
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.cam)
        return self.renderer.render()


def humanoid_angles(targets):
    """[cx, cy, size, grip] (hand in the camera image) -> [shoulder, arm, elbow, gripper] for the humanoid's right arm.

    Mirror-style: hand up -> arm raised, hand moves away from the body (image left for a right hand facing the camera) ->
    arm swings outward, hand close to the camera -> arm straight, far -> elbow bent, fist -> gripper closed."""
    cx, cy, size, grip = np.asarray(targets, float)[:4]
    lin = lambda x, s, d: d[0] + np.clip((x - s[0]) / (s[1] - s[0]), 0, 1) * (d[1] - d[0])
    return np.array([lin(cy, (0.9, 0.1), (-0.2, 1.4)), lin(cx, (0.1, 0.9), (0.9, -0.25)),
                     lin(size, (0.03, 0.15), (0.9, 0.0)), lin(grip, (0, 1), (0, 1.39))])


class ArmIK:
    """Fit the humanoid's right-arm joints [shoulder, arm, elbow] so that, seen from the front, the robot's upper arm and forearm
    point the same way as the person's (2D unit vectors in the 'outward, up' plane). Damped Gauss-Newton, warm-started."""
    LIMITS = np.array([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
    REST = np.array([0.0, -0.38, 0.0])                       # arm hanging straight down

    def __init__(self, robot: "Humanoid"):
        self.r = robot
        m = robot.model
        self.j = [m.joint(n).id for n in ("right_shoulder_revolute", "right_elbow_revolute", "left_hand_tip_revolute")]
        self.q = self.REST.copy(); self.clipped = False

    def _dirs(self, q):
        r = self.r
        r.data.qpos[:] = r.q0
        for a, v in zip(r.adr[:3], q):
            r.data.qpos[a] = v
        mujoco.mj_forward(r.model, r.data)
        s, e, w = (r.data.xanchor[i] for i in self.j)
        out = lambda d: np.array([-d[0], d[2]]) / (np.linalg.norm([d[0], d[2]]) + 1e-9)   # (outward, up); this limb is at x<0
        return np.concatenate([out(e - s), out(w - e)])

    def fit(self, human_u, human_f, iters=12):
        """human_u/f: 2D vectors (outward, up) of the person's upper arm / forearm. Returns (q [3], residual angle deg upper/fore)."""
        n = lambda v: np.asarray(v, float) / (np.linalg.norm(v) + 1e-9)
        # Range-limited retarget: the robot's shoulder reaches ~108 deg from straight down and its elbow +-86 deg, a person's arm goes to 180.
        # Compress elevation above 90 deg into 90..108 and clip the elbow flexion, so the target is reachable (keeps the pose's look).
        ang = lambda v: np.degrees(np.arctan2(v[0], -v[1]))                  # 0 = down, 90 = outward horizontal, 180 = up
        au, af = ang(human_u), ang(human_f)
        rel = (af - au + 180) % 360 - 180
        au_c = au if au <= 90 else 90 + (au - 90) * 0.2
        au_c = max(au_c, -60)
        rel_c = np.clip(rel, -86, 86)
        self.clipped = abs(rel) > 86 or au > 108
        vec = lambda a: np.array([np.sin(np.radians(a)), -np.cos(np.radians(a))])
        target = np.concatenate([vec(au_c), vec(au_c + rel_c)])
        best = None
        for q0 in (self.q, self.REST):
            q = q0.copy()
            for _ in range(iters):
                f0 = self._dirs(q) - target
                J = np.stack([(self._dirs(q + np.eye(3)[i] * 1e-3) - self._dirs(q)) / 1e-3 for i in range(3)], 1)
                reg = np.diag([0.05, 0.02, 0.02])                 # stay near rest unless the image demands otherwise
                dq = np.linalg.solve(J.T @ J + reg + 1e-3 * np.eye(3), -(J.T @ f0 + reg @ (q - self.REST)))
                q = np.clip(q + dq, self.LIMITS[:, 0], self.LIMITS[:, 1])
            c = np.linalg.norm(self._dirs(q) - target)
            if best is None or c < best[0]:
                best = (c, q)
        self.q = best[1]
        d = self._dirs(self.q)
        ang = lambda a, b: np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))
        return self.q.copy(), (ang(d[:2], target[:2]), ang(d[2:], target[2:]))


class ArmIK3(ArmIK):
    """ArmIK with a depth cue: targets are 3D unit vectors (outward, up, forward). The forward part comes from foreshortening, which a single
    front camera can only give as a magnitude (it cannot tell towards-camera from away, so forward is assumed: reaching toward the camera)."""

    def __init__(self, robot):
        super().__init__(robot)
        d0 = self._dirs(self.REST); d1 = self._dirs(np.array([0.5, self.REST[1], 0.0]))
        self.fwd = 1.0 if d1[2] - d0[2] >= 0 else -1.0          # sign of world-y that corresponds to 'forward' (positive shoulder pitch raises forward)

    def _dirs(self, q):
        r = self.r
        r.data.qpos[:] = r.q0
        for a, v in zip(r.adr[:3], q):
            r.data.qpos[a] = v
        mujoco.mj_forward(r.model, r.data)
        s, e, w = (r.data.xanchor[i] for i in self.j)
        f = getattr(self, "fwd", 1.0)
        nrm = lambda d: np.array([-d[0], d[2], f * d[1]]) / (np.linalg.norm(d) + 1e-9)
        return np.concatenate([nrm(e - s), nrm(w - e)])      # (outward, up, forward) x2

    def fit(self, human_u, human_f, fu=0.0, ff=0.0, iters=12):
        """human_u/f: 2D (outward, up) vectors; fu/ff in [0,1]: forward fraction of the upper arm / forearm (from foreshortening)."""
        ang = lambda v: np.degrees(np.arctan2(v[0], -v[1]))
        vec = lambda a: np.array([np.sin(np.radians(a)), -np.cos(np.radians(a))])
        au, af = ang(human_u), ang(human_f)
        rel = np.clip((af - au + 180) % 360 - 180, -86, 86)
        au_c = max(au if au <= 90 else 90 + (au - 90) * 0.2, -60)
        self.clipped = abs(af - au) > 86 or au > 108
        mk = lambda a, fr: np.array([*(vec(a) * np.sqrt(max(0.0, 1 - fr * fr))), fr])
        target = np.concatenate([mk(au_c, fu), mk(au_c + rel, ff)])
        best = None
        for q0 in (self.q, self.REST):
            q = q0.copy()
            for _ in range(iters):
                f0 = self._dirs(q) - target
                J = np.stack([(self._dirs(q + np.eye(3)[i] * 1e-3) - self._dirs(q)) / 1e-3 for i in range(3)], 1)
                reg = np.diag([0.02, 0.02, 0.02])
                dq = np.linalg.solve(J.T @ J + reg + 1e-3 * np.eye(3), -(J.T @ f0 + reg @ (q - self.REST)))
                q = np.clip(q + dq, self.LIMITS[:, 0], self.LIMITS[:, 1])
            c = np.linalg.norm(self._dirs(q) - target)
            if best is None or c < best[0]: best = (c, q)
        self.q = best[1]; d = self._dirs(self.q)
        a3 = lambda a, b: np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))
        return self.q.copy(), (a3(d[:3], target[:3]), a3(d[3:], target[3:]))
