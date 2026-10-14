"""Choose a headless renderer without launching training or legacy tasks."""
from pathlib import Path

import numpy as np


def main():
    import mujoco

    root = Path(__file__).resolve().parents[1]
    scene = root / "envs/mjx_walking/assets/humanoid_rubber_2mm_servo_v2_kp40.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "standing_start")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    try:
        renderer = mujoco.Renderer(model, height=240, width=320)
        try:
            renderer.update_scene(data)
            frame = renderer.render()
            if frame.shape != (240, 320, 3) or not np.isfinite(frame).all() or frame.max() == frame.min():
                raise RuntimeError("The renderer returned a blank or malformed image")
        finally:
            renderer.close()
    except Exception as exc:
        print(f"Rendering failed: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(3) from exc
    print("Headless render smoke passed; this is not GPU physics or walking qualification.")


if __name__ == "__main__":
    main()
