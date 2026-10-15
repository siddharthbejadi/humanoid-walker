"""Draw the README figures from results/measurements.json.   python tools/make_figures.py   (needs matplotlib)
Colours: reference palette slots 1 (blue) and 2 (orange); validated with the dataviz validator (contrast >= 3:1, CVD dE 24.7, normal dE 33.6).
Rules applied: one axis, thin marks, quiet grid, direct value labels, legend for two series, text in text colours, never the series colour."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent
M = json.loads((R / "results" / "measurements.json").read_text())
SURFACE, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def base(w=8.4, h=4.4, title="", sub=""):
    fig, ax = plt.subplots(figsize=(w, h))
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.text(0.02, 0.965, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.905, sub, fontsize=10, color=INK2, va="top")
    return fig, ax


# 1) arm found by camera view (single series -> no legend; the title names it)
v = sorted(M["arm_found_by_camera_view"]["views"], key=lambda d: d["pct"])
fig, ax = base(title="Where the camera sits decides how often the arm is found", sub="Share of frames where MoveNet's right arm passes the plausibility gate (higher is better)")
ys = range(len(v)); ax.barh(list(ys), [d["pct"] for d in v], height=0.5, color=BLUE)
ax.set_yticks(list(ys)); ax.set_yticklabels([f'{d["label"]}\n({d["frames"]} frames)' for d in v], fontsize=9); ax.set_xlim(0, 112); ax.set_xlabel("frames with the arm found (%)")
for y, d in zip(ys, v): ax.text(d["pct"] + 1.5, y, f'{d["pct"]:.1f}%', va="center", color=INK, fontsize=11)
fig.subplots_adjust(left=0.42, right=0.97, top=0.80, bottom=0.14); fig.savefig(R / "figures" / "fig_arm_found_by_view.png", dpi=160); plt.close(fig)

# 2) smoothing trade-off: motion left vs lag (single series; points labelled directly)
s = M["smoothing_one_euro"]["settings"]; fps = 15.0
fig, ax = base(title="Smoothing trades jitter for lag", sub="One-Euro filter on the IK joint angles; raw motion = 100%. Lower and further left is better")
xs = [d["lag_frames"] / fps * 1000 for d in s]; ys_ = [d["motion_left_pct"] for d in s]
ax.plot(xs, ys_, color=BLUE, lw=1.6, zorder=2); ax.scatter(xs, ys_, s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
for x, y, d in zip(xs, ys_, s): ax.annotate(f'cut-off {d["cutoff_hz"]} Hz: {d["motion_left_pct"]}% left, {d["lag_frames"]} frame lag', (x, y), textcoords="offset points", xytext=(10, 8), fontsize=10, color=INK)
ax.set_xlabel("lag (ms, at 15 fps)"); ax.set_ylabel("motion left (% of raw)"); ax.set_ylim(0, 60); ax.set_xlim(-10, 260); ax.grid(axis="y", color=GRID, lw=0.8)
fig.subplots_adjust(left=0.1, right=0.97, top=0.80, bottom=0.15); fig.savefig(R / "figures" / "fig_smoothing_tradeoff.png", dpi=160); plt.close(fig)

# 3) learned student vs mean-pose baseline (two series -> legend + direct labels)
a, b = M["learned_student_vs_mean_pose"]["first_student_same_session_val"], M["learned_student_vs_mean_pose"]["final_student_take3_heldout_tail"]
groups = ["First student\n(two clips, last 20%)", "Final student\n(take 3, held-out end)"]; stu, base_ = [a["student"], b["student"]], [a["baseline"], b["baseline"]]
fig, ax = base(w=8.4, h=4.6, title="The learned network beat a constant guess once, then did not", sub="Median upper-arm direction error seen from the front, degrees (lower is better)")
ax.grid(axis="x", visible=False); ax.grid(axis="y", color=GRID, lw=0.8)
w = 0.32; xs = [0, 1]
b1 = ax.bar([x - w / 2 - 0.01 for x in xs], stu, w, color=BLUE, label="Student"); b2 = ax.bar([x + w / 2 + 0.01 for x in xs], base_, w, color=ORANGE, label="Mean-pose baseline")
for bars in (b1, b2):
    for r in bars: ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.5, f"{r.get_height():.1f}°", ha="center", color=INK, fontsize=11)
ax.set_xticks(xs); ax.set_xticklabels(groups); ax.set_ylabel("degrees"); ax.set_ylim(0, 27); ax.legend(frameon=False, loc="upper right")
fig.text(0.02, 0.075, "The baseline is the test frames' own mean pose, a strong constant guess.", fontsize=8.5, color=INK2)
fig.text(0.02, 0.025, "On an unseen camera setup the network failed completely (91° error).", fontsize=8.5, color=INK2)
fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.26); fig.savefig(R / "figures" / "fig_student_vs_baseline.png", dpi=160); plt.close(fig)

# 4) gripper on FreiHAND (model vs majority-class baseline)
g = M["gripper_freihand"]; labels = ["Hand at normal size", "Hand shrunk to 24 px", "Always guess the\nmajority class"]; vals = [g["close"] * 100, g["far_24px"] * 100, g["majority_class_baseline"] * 100]
fig, ax = base(w=8.4, h=4.4, title="Gripper network: open vs closed hand on FreiHAND", sub="Accuracy on 900 held-out FreiHAND images (not evidence for the author's own hand)")
ax.grid(axis="x", visible=False); ax.grid(axis="y", color=GRID, lw=0.8)
cols = [BLUE, BLUE, ORANGE]; bars = ax.bar(range(3), vals, 0.5, color=cols)
for r, vv in zip(bars, vals): ax.text(r.get_x() + r.get_width() / 2, vv + 1.2, f"{vv:.1f}%", ha="center", color=INK, fontsize=11)
ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.set_ylim(0, 105); ax.set_ylabel("accuracy (%)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="Gripper network"), Patch(color=ORANGE, label="Baseline")], frameon=False, loc="upper right")
fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.14); fig.savefig(R / "figures" / "fig_gripper_freihand.png", dpi=160); plt.close(fig)
print("wrote", sorted(p.name for p in (R / "figures").glob("*.png")))
