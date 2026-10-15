"""One results dashboard: six small panels, drawn from results/measurements.json.   python eval/make_dashboard.py   (needs matplotlib)
Design rules (dataviz method): one idea per panel with the finding as its title, one axis each, thin marks, quiet grid, direct value labels, a legend only where two series share a panel,
text in text colours (never the series colour), palette = reference slots 1 (blue) and 2 (orange), validated for contrast >= 3:1 and colour-blind separation (dE 24.7)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

R = Path(__file__).resolve().parent.parent
M = json.loads((R / "results" / "measurements.json").read_text())
SURFACE, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.6)); axs = axs.ravel()


def panel(ax, title, sub, grid="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=24); ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")


def labels(ax, bars, fmt, dy):
    for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt(b.get_height()), ha="center", color=INK, fontsize=9.5)


# 1) camera view -> arm found
v = sorted(M["arm_found_by_camera_view"]["views"], key=lambda d: d["pct"]); ax = axs[0]
panel(ax, "Camera placement decides coverage", "Frames where the arm passes the gate (%)", grid="x")
bars = ax.barh(range(len(v)), [d["pct"] for d in v], height=0.5, color=BLUE); ax.set_yticks(range(len(v)))
SHORT = {"Level view, whole body (take 3)": "Level, whole body\n(take 3)", "Upright view, live recording": "Upright\n(live recording)", "Upright view, forward-raise take (incl. setup)": "Upright\n(forward-raise take)", "Low, tilted view (side-raise take)": "Low and tilted\n(side-raise take)"}
ax.set_yticklabels([SHORT.get(d["label"], d["label"]) for d in v], fontsize=8.5); ax.set_xlim(0, 118)
for y, d in enumerate(v): ax.text(d["pct"] + 1.5, y, f'{d["pct"]:.1f}%', va="center", color=INK, fontsize=9.5)

# 2) gate threshold sweep
g = M["gate_threshold_sweep"]["points"]; ax = axs[1]
panel(ax, "A lower score threshold buys coverage", "Frames passing the gate vs minimum keypoint score; 0.04 chosen")
xs = [p["min_score"] for p in g]; ys = [p["pct"] for p in g]
ax.plot(xs, ys, color=BLUE, lw=1.6, zorder=2); ax.scatter(xs, ys, s=46, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
for x, y in zip(xs, ys): ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9.5, color=INK)
ax.axvline(0.04, color=INK2, lw=0.9, ls=(0, (3, 3))); ax.text(0.0415, 38, "chosen", fontsize=8.5, color=INK2, ha="right"); ax.invert_xaxis(); ax.set_xticks(xs); ax.set_xticklabels([f"{x:.2f}" for x in xs]); ax.set_ylim(35, 82); ax.set_xlabel("minimum keypoint score (stricter on the left)"); ax.set_ylabel("%")
ax.text(0.5, -0.30, "accuracy at lower thresholds not measured", transform=ax.transAxes, fontsize=8, color=INK2, ha="center")

# 3) smoothing trade-off
s = M["smoothing_one_euro"]["settings"]; ax = axs[2]
panel(ax, "Smoothing trades jitter for lag", "Joint-angle motion left after the One-Euro filter (raw = 100%)", grid="y")
xs = [d["lag_frames"] / 15 * 1000 for d in s]; ys = [d["motion_left_pct"] for d in s]
ax.plot(xs, ys, color=BLUE, lw=1.6); ax.scatter(xs, ys, s=46, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
for x, y, d in zip(xs, ys, s): ax.annotate(f'{d["cutoff_hz"]} Hz: {y}%', (x, y), textcoords="offset points", xytext=(8, 7), fontsize=9, color=INK)
ax.set_xlim(-12, 215); ax.set_ylim(0, 56); ax.set_xlabel("lag (ms at 15 fps)"); ax.set_ylabel("motion left (%)")

# 4) fine-tune vs original
f = M["movenet_finetune_vs_original"]["unseen_live_take"]; ax = axs[3]
panel(ax, "Fine-tuning: smoother, not more accurate", f"Jitter in px per frame, lower is better. Unseen take; coverage {f['coverage_pct']['original']}% vs {f['coverage_pct']['finetuned']}%")
bars = ax.bar([0, 1], [f["jitter_px_per_frame"]["original"], f["jitter_px_per_frame"]["finetuned"]], 0.45, color=[ORANGE, BLUE]); labels(ax, bars, lambda h: f"{h:.1f}", 0.25)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Original zoo\nMoveNet", "Fine-tuned"]); ax.set_ylim(0, 14); ax.set_ylabel("px / frame")

# 5) learned network vs baseline (two series: legend)
L = M["learned_student_vs_mean_pose"]; a_, b_ = L["first_student_same_session_val"], L["final_student_take3_heldout_tail"]; ax = axs[4]
panel(ax, "A learned network beat a constant guess once", "Median upper-arm direction error, degrees (lower is better)")
w = 0.34; b1 = ax.bar([-w / 2 - 0.01, 1 - w / 2 - 0.01], [a_["student"], b_["student"]], w, color=BLUE); b2 = ax.bar([w / 2 + 0.01, 1 + w / 2 + 0.01], [a_["baseline"], b_["baseline"]], w, color=ORANGE)
labels(ax, list(b1) + list(b2), lambda h: f"{h:.1f}°", 0.5); ax.set_xticks([0, 1]); ax.set_xticklabels(["First student\n(last 20% of 2 clips)", "Final student\n(held-out end of take 3)"]); ax.set_ylim(0, 28)
ax.legend(handles=[Patch(color=BLUE, label="Student"), Patch(color=ORANGE, label="Mean-pose baseline")], frameon=False, loc="upper right", fontsize=8.5)
ax.text(0.5, -0.30, "on an unseen camera setup it failed completely (91°)", transform=ax.transAxes, fontsize=8, color=INK2, ha="center")

# 6) gripper
G = M["gripper_freihand"]; ax = axs[5]
panel(ax, "Gripper works on FreiHAND, not on my footage", "Open/closed accuracy on 900 held-out FreiHAND images (%)")
vals = [G["close"] * 100, G["far_24px"] * 100, G["majority_class_baseline"] * 100]; bars = ax.bar(range(3), vals, 0.5, color=[BLUE, BLUE, ORANGE]); labels(ax, bars, lambda h: f"{h:.1f}%", 1.4)
ax.set_xticks(range(3)); ax.set_xticklabels(["Normal size", "Shrunk to\n24 px", "Majority-class\nguess"]); ax.set_ylim(0, 108)
ax.legend(handles=[Patch(color=BLUE, label="Gripper network"), Patch(color=ORANGE, label="Baseline")], frameon=False, loc="upper right", fontsize=8.5)

fig.suptitle("Manipulation results at a glance: every bar is measured, including the ones that did not work", x=0.01, ha="left", fontsize=13.5, fontweight="bold", color=INK)
fig.subplots_adjust(left=0.105, right=0.985, top=0.84, bottom=0.12, wspace=0.34, hspace=0.62)
out = R / "figures" / "fig_dashboard.png"; fig.savefig(out, dpi=150); print("wrote", out)
