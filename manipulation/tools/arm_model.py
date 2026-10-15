"""A small 4-joint arm (shoulder pitch, sideways swing, elbow, gripper) modelled on the humanoid's servo arm.

Kinematic model for MuJoCo: two copies share one base, the solid copy is driven by the student, the translucent
"ghost" copy by the teacher (hand tracking), so tracking error is visible. Joint angles come from hc_common.retarget().
"""
import mujoco

UPPER, FORE, HAND = 0.10, 0.11, 0.05          # link lengths in metres (servo-arm sized)


def _chain(prefix, rgba, alpha):
    r = " ".join(str(v) for v in (*rgba, alpha))
    return f"""
      <body name="{prefix}shoulder" pos="0 0 0">
        <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
        <joint name="{prefix}shoulder" type="hinge" axis="1 0 0" range="-1.7 1.7"/>
        <body name="{prefix}arm" pos="0 0 0">
        <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
          <joint name="{prefix}arm" type="hinge" axis="0 1 0" range="-1.7 1.7"/>
          <geom type="capsule" fromto="0 0 0 0 0 -{UPPER}" size="0.012" rgba="{r}" contype="0" conaffinity="0"/>
          <body name="{prefix}forearm" pos="0 0 -{UPPER}">
        <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
            <joint name="{prefix}elbow" type="hinge" axis="1 0 0" range="-0.2 1.7"/>
            <geom type="capsule" fromto="0 0 0 0 0 -{FORE}" size="0.010" rgba="{r}" contype="0" conaffinity="0"/>
            <body name="{prefix}hand" pos="0 0 -{FORE}">
        <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
              <geom type="box" size="0.02 0.008 0.02" pos="0 0 -0.02" rgba="{r}" contype="0" conaffinity="0"/>
              <body name="{prefix}finger" pos="0 0.012 -0.04">
        <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
                <joint name="{prefix}grip" type="hinge" axis="1 0 0" range="0 1.5"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.04" size="0.005" rgba="{r}" contype="0" conaffinity="0"/>
                <site name="{prefix}tip" pos="0 0 -0.04" size="0.006"/>
              </body>
            </body>
          </body>
        </body>
      </body>"""


def build_model():
    xml = f"""
<mujoco model="rtas_arm">
  <visual><global offwidth="640" offheight="640"/></visual>
  <worldbody>
    <light pos="0.4 -0.6 0.8" dir="-0.3 0.6 -0.7"/>
    <geom name="floor" type="plane" pos="0 0 -0.35" size="0.6 0.6 0.1" rgba="0.92 0.92 0.92 1"/>
    <body name="torso" pos="0 0 0">
      <geom type="box" size="0.07 0.04 0.12" pos="0.0 0 0.10" rgba="0.25 0.25 0.28 1" contype="0" conaffinity="0"/>
      {_chain("student_", (0.1, 0.35, 0.95), 1.0)}
      {_chain("teacher_", (0.1, 0.8, 0.2), 0.35)}
    </body>
  </worldbody>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


JOINTS = ["shoulder", "arm", "elbow", "grip"]


def set_pose(model, data, prefix, angles):
    """angles: (4,) [shoulder, arm, elbow, gripper] rad. Returns the fingertip position (m)."""
    for name, a in zip(JOINTS, angles):
        data.qpos[model.joint(prefix + name).qposadr[0]] = a
    mujoco.mj_forward(model, data)
    return data.site_xpos[model.site(prefix + "tip").id].copy()
