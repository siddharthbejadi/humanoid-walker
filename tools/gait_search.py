"""CEM search for a scripted-gait variant that walks under real-robot limits (torque, slew, delay, IMU noise).
One candidate per world; all candidates share the same physics settings given by overrides."""
import os, sys, json, time
from pathlib import Path
ROOT = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(ROOT))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false"); os.environ.setdefault("JAX_PLATFORMS", "cuda")
import numpy as np, hydra
from scripts.collect_script_free_demonstrations import GPUStudentProvider
from scripted_expert import ScriptedExpert
out_path = Path(sys.argv[2]); overrides = json.loads(sys.argv[3]); gens = int(sys.argv[4]); delay = int(sys.argv[5]); noise = float(sys.argv[6])
W = int(os.environ.get("POP", 32)); STEPS = 1000
with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
    config = hydra.compose("walking_mjx_script_free", overrides=["seed=100"] + overrides)
config.env.env_num = W; config.env.eval_episode_num = 0
prov = GPUStudentProvider(config, W)
ref = ScriptedExpert.from_npz(ROOT / "envs/mjx_walking/assets/humanlike_gait.npz")
C, cs, L = ref.ctrl, ref.cycle_start, ref.cycle_len
cmean = C[cs:cs + L].mean(0)
# params: time scale s, leg amplitude a, kp, kd, limit
NAMES = ["s", "a", "kp", "kd", "lim"]
LO = np.array([.5, .5, .3, 0., .1]); HI = np.array([1., 1.1, 2.5, .12, .4])
def targets(t, P, roll, rate_state):
    s, a, kp, kd, lim = (P[:, i] for i in range(5))
    phase = np.where(t < cs, t, cs + s * (t - cs))
    base = np.zeros((len(P), 19), np.float32)
    for w in range(len(P)):
        ph = phase[w]
        if ph < cs:
            i0 = int(ph); i1 = min(i0 + 1, len(C) - 1); fr = ph - i0
        else:
            q = (ph - cs) % L; i0 = cs + int(q); i1 = cs + (int(q) + 1) % L; fr = q - int(q)
        base[w] = (1 - fr) * C[i0] + fr * C[i1]
    blend = np.clip((t - cs) / 40., 0., 1.)
    amp = 1. + (a - 1.) * blend
    base[:, :10] = cmean[:10] + amp[:, None] * (base[:, :10] - cmean[:10]) if t >= cs else base[:, :10]
    prev, filt = rate_state
    filt = .8 * filt + .2 * (roll - prev) / .02
    corr = np.clip(kp * roll + kd * filt, -lim, lim)
    base[:, 0] += -corr; base[:, 5] += -corr; base[:, 4] += corr; base[:, 9] -= corr
    return base, (roll.copy(), filt)
rng = np.random.default_rng(1)
mu = np.array([.85, .9, 1., .04, .25]); sd = (HI - LO) * .25
best = None; log = []
for g in range(gens):
    P = np.clip(rng.normal(mu, sd, (W, 5)), LO, HI)
    if g == 0: P[0] = [1., 1., 1., .04, .25]   # the original gait
    first = prov.reset(100 + g); audit = first["audit"]
    roll0 = ScriptedExpert.roll(audit["rotation"]); st = (roll0.copy(), np.zeros(W))
    alive = np.full(W, STEPS); dead = np.zeros(W, bool); valid = np.zeros(W); alt = np.zeros(W); tilt = np.zeros(W)
    y0 = audit["qpos"][:, 1].copy(); y = y0.copy(); queue = []
    for t in range(STEPS):
        roll = ScriptedExpert.roll(audit["rotation"]) + rng.normal(0, noise, W)
        tg, st = targets(t, P, roll, st); queue.append(tg)
        out = prov.step(prov.normalize(queue[max(0, len(queue) - 1 - delay)])); audit = out["audit"]
        act = ~dead
        lg = out["logs"]
        valid += act * (np.asarray(lg["valid_left_step"]).reshape(W) + np.asarray(lg["valid_right_step"]).reshape(W))
        alt += act * np.asarray(lg["alternating_steps"]).reshape(W)
        tilt = np.maximum(tilt, act * np.asarray(lg["body_tilt_rad_peak"]).reshape(W))
        y = np.where(act, audit["qpos"][:, 1], y)
        newly = out["ended"] & ~dead; alive[newly] = t + 1; dead |= out["ended"]
        if dead.all(): break
    fall = np.asarray(out["causes"]["physical_fall"]).reshape(W)  # last-step cause only; use alive<STEPS as fall proxy
    fwd = -(y - y0)
    score = 10 * alive / STEPS + 5 * np.clip(fwd, 0, 1.2) / 1.2 + 5 * np.clip(valid, 0, 40) / 40 + 3 * np.clip(alt, 0, 40) / 40 - 2 * np.degrees(tilt) / 45
    order = np.argsort(-score); elite = P[order[:max(4, W // 4)]]
    mu = elite.mean(0); sd = np.maximum(elite.std(0), (HI - LO) * .03)
    b = order[0]
    rec = dict(gen=g, best_score=round(float(score[b]), 2), alive=int(alive[b]), fwd=round(float(fwd[b]), 2), valid=float(valid[b]), alt=float(alt[b]),
               tilt_deg=round(float(np.degrees(tilt[b])), 1), n_full=int((alive == STEPS).sum()), params={n: round(float(P[b, i]), 3) for i, n in enumerate(NAMES)})
    print(json.dumps(rec), flush=True); log.append(rec)
    if best is None or score[b] > best[0]: best = (float(score[b]), P[b].tolist(), rec)
    out_path.write_text(json.dumps(dict(best_score=best[0], best_params=dict(zip(NAMES, best[1])), best_record=best[2], log=log), indent=1))
prov.close()
