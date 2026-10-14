"""Expert robustness under real-robot-like limits. variants: {name, overrides[], delay, roll_noise}"""
import os, sys, json
from pathlib import Path
ROOT = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false"); os.environ.setdefault("JAX_PLATFORMS", "cuda")
import numpy as np, hydra
from scripts.collect_script_free_demonstrations import GPUStudentProvider
from scripted_expert import ScriptedExpert
variants = json.loads(sys.argv[2]); steps = int(sys.argv[3])
W = 4
for v in variants:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = hydra.compose("walking_mjx_script_free", overrides=["seed=100"] + v.get("overrides", []))
    config.env.env_num = W; config.env.eval_episode_num = 0
    p = GPUStudentProvider(config, W)
    kp, kd, lim = v.get("stab", [1., .04, .25]); ex = ScriptedExpert.from_npz(ROOT / "envs/mjx_walking/assets/humanlike_gait.npz", stabilizer=(kp, kd, -1., lim), time_scale=v.get("ts", 1.), amplitude=v.get("amp", 1.))
    rng = np.random.default_rng(0)
    first = p.reset(100); audit = first["audit"]; ex.reset(audit["rotation"])
    delay = int(v.get("delay", 0)); queue = []
    acc = {}; ended_at = None; y0 = audit["qpos"][:, 1].copy(); tilt_peak = 0.
    for t in range(steps):
        rot = audit["rotation"].copy()
        d = np.asarray(ex.targets(t, rot), np.float32)
        queue.append(d)
        use = queue[max(0, len(queue) - 1 - delay)]
        out = p.step(p.normalize(use)); audit = out["audit"]
        for k in ('valid_left_step','valid_right_step','alternating_steps'):
            acc[k] = acc.get(k, 0) + np.asarray(out['logs'][k]).reshape(W)
        tilt_peak = max(tilt_peak, float(np.asarray(out['logs']['body_tilt_rad_peak']).max()))
        if out["ended"].any():
            ended_at = t; break
    cause = {k: bool(np.asarray(vv).any()) for k, vv in out["causes"].items()}
    print(json.dumps(dict(name=v["name"], steps_run=t + 1, ended_at=ended_at, cause=cause,
        fwd_m=[round(float(-(audit["qpos"][i, 1] - y0[i])), 2) for i in range(W)],
        steps={k: round(float(np.mean(vv)), 1) for k, vv in acc.items()}, tilt_peak_deg=round(float(np.degrees(tilt_peak)), 1))), flush=True)
    p.close()
