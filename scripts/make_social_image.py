"""Tweet image: stated confidence when the true answer is a coin flip.

Numbers are the dept_double.mean_max_p field of each run's eval/probe.json,
i.e. the published checkpoint (runs/q-rlcd) and its two comparison arms.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
HERO = "#2a78d6"
MUTED = "#9a9992"
IDEAL = 0.5

runs = [
    ("outcome reward\n(RLVR)", "runs/q-rlvr-lowlr", MUTED),
    ("trained on the\nlabels", "runs/q-oracle", MUTED),
    ("calibration reward\n(RLCD)", "runs/q-rlcd", HERO),
]
bars = []
for label, run, color in runs:
    report = json.load(open(f"{run}/eval/probe.json", encoding="utf-8"))["report"]
    bars.append((label, report["dept_double"]["mean_max_p"], color, report["dept_double"]["n"]))

fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

xs = list(range(len(bars)))
ax.bar(xs, [b[1] for b in bars], width=0.54, color=[b[2] for b in bars], zorder=3)
ax.axhline(IDEAL, color=INK, linestyle=(0, (5, 4)), linewidth=2, zorder=4)

for x, (_, value, color, _n) in zip(xs, bars):
    ax.text(x, value + 0.025, f"{value:.2f}", ha="center", va="bottom",
            fontsize=21, fontweight="bold", color=INK if color == HERO else INK_2)

ax.set_xticks(xs)
ax.set_xticklabels([b[0] for b in bars], fontsize=13.5, color=INK_2)
ax.set_ylim(0, 1.18)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"], fontsize=11, color=INK_2)
ax.set_ylabel("stated confidence", fontsize=12.5, color=INK_2, labelpad=8)
ax.grid(axis="y", color="#e6e5df", linewidth=1, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#d8d7d1")
ax.tick_params(length=0)

fig.text(0.035, 0.955, "When the true answer is a coin flip",
         fontsize=19, fontweight="bold", color=INK, va="top")
fig.text(0.035, 0.895,
         "Dashed line is the truth. Same 0.6B model, same warmup, same training loop.",
         fontsize=11.5, color=INK_2, va="top")
fig.text(0.035, 0.028,
         f"Qwen3-0.6B-Base on {bars[0][3]} held-out tickets built with a 0.5/0.5 answer.\n"
         "The two reward arms see only whether the option they picked was right.",
         fontsize=9.5, color=INK_2, va="bottom", linespacing=1.5)

fig.subplots_adjust(left=0.10, right=0.975, top=0.845, bottom=0.215)
out = "docs/img/tweet-coinflip.png"
fig.savefig(out, facecolor=SURFACE)
print("wrote", out, {b[0].replace(chr(10), " "): round(b[1], 4) for b in bars})
