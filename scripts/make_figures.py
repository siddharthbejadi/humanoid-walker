"""Regenerate figures/*.png from results/*.json.  Usage: python scripts/make_figures.py  (needs matplotlib)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R, OUT = ROOT / "results", ROOT / "figures"
OUT.mkdir(exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"      # categorical slots 1, 2
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID})


def clean(ax, grid="y"):
    ax.grid(axis=grid, color=GRID, lw=.8); ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight"); plt.close(fig)


load = lambda p: json.loads((R / p).read_text())

# 1. behaviour-cloning validation error, before vs after
runs = [("5k updates", load("pretraining/bc5k.json")), ("12k updates (deployed)", load("pretraining/bc12k_realistic.json"))]
fig, ax = plt.subplots(figsize=(6.4, 3.4)); clean(ax)
w = .36
for i, (n, d) in enumerate(runs):
    for j, (v, c, lab) in enumerate([(d["initial_validation_bc"], BLUE, "before training"), (d["full_episode_test"]["bc_mse"], ORANGE, "after training (held-out test)")]):
        ax.bar(i + (j - .5) * w * 1.05, v, w, color=c, label=lab if i == 0 else None)
        ax.text(i + (j - .5) * w * 1.05, v * 1.25, f"{v:.1e}", ha="center", fontsize=9, color=INK)
ax.set_yscale("log"); ax.set_ylim(1e-6, 1e1)
ax.set_xticks(range(2)); ax.set_xticklabels([n for n, _ in runs]); ax.set_ylabel("action MSE (log scale)")
ax.set_title("Behaviour cloning: error on unseen episodes drops ~6000x", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=9)
save(fig, "fig_bc_error.png")

# 2. closed-loop evaluation, 3 seeds, 32 simulated robots each, 20 s standing start
ev = {n: load(f"evals/real_{n}.json")["summary"] for n in (1100, 1200, 1300)}
fig, ax = plt.subplots(figsize=(6.4, 3.4)); clean(ax)
for i, (n, s) in enumerate(ev.items()):
    ax.bar(i, s["forward_distance"], .5, color=BLUE)
    ax.text(i, s["forward_distance"] + .03, f'{s["forward_distance"]:.2f} m\n{s["total_valid_steps"]:.0f} valid steps', ha="center", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels([f"eval seed {n}" for n in ev]); ax.set_ylim(0, 1.9)
ax.set_ylabel("forward distance in 20 s (mean of 32 robots)")
ax.set_title("Script-free student, closed loop in simulation", loc="left", fontsize=11, color=INK)
save(fig, "fig_closed_loop_eval.png")

# 3. gait search: best score per generation, full servo torque vs measured limits
fig, ax = plt.subplots(figsize=(6.4, 3.4)); clean(ax)
for f, c, lab in (("gait_search/search_18kg.json", BLUE, "18 kg·cm torque (sim default)"), ("gait_search/search_real.json", ORANGE, "measured servo limits")):
    d = load(f); g = [l["gen"] for l in d["log"]]; s = [l["best_score"] for l in d["log"]]
    ax.plot(g, s, color=c, lw=2, marker="o", ms=4, mfc=SURF, mew=1.5)
    ax.text(g[-1] + .2, s[-1], lab, color=INK, va="center", fontsize=9)
    ax.text(g[-1] + .2, s[-1] - 1.3, f'{d["best_record"]["valid"]:.0f} valid steps', color=INK2, va="center", fontsize=8.5)
ax.set_xlim(0, 19); ax.set_xticks(range(0, 14, 2)); ax.set_xlabel("search generation"); ax.set_ylabel("best score")
ax.set_title("Gait search finds a walk at full torque, not at measured limits", loc="left", fontsize=11, color=INK)
save(fig, "fig_gait_search.png")

# 4. stress tests of the scripted walk (numbers from docs/PROJECT_NOTES_12Oct.md, items 3 and 5)
fig, ax = plt.subplots(figsize=(6.4, 2.8)); clean(ax, "x")
rows = [("all real limits combined", 4.6, ORANGE, "fell at 4.6 s"), ("0.9 N·m on every joint", 11, ORANGE, "fell at 11 s"), ("1.77 N·m (sim default)", 84, BLUE, "ran full 84 s, no fall")]
for i, (n, v, c, t) in enumerate(rows):
    ax.barh(i, v, .5, color=c); ax.text(v + 1.5, i, t, va="center", fontsize=9)
ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0, 110); ax.set_xlabel("seconds upright (CPU sim, scripted walk)")
ax.set_title("Real servo limits break the scripted walk", loc="left", fontsize=11, color=INK)
save(fig, "fig_stress.png")
print("ok")
