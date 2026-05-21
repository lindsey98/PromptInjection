import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator, PercentFormatter

# --- Font and style settings ---
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": ["sans-serif"],
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.5,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

PASTEL_PALETTE = [
    "#95bde5",  # light blue
    "#4c648b",  # deep blue
    "#e7a8a1",  # soft red/pink
    "#c1d4a4",  # muted green
    "#789262",  # darker green
    "#f3c999",  # peach
    "#a47d8e",  # brown-purple
    "#875c5c",  # darker red
    "#c2b7d2",  # light purple
    "#000000",  # black for total
]
def plot_grouped_bars_ax(ax, categories, series_dict,
                         ylabel=None, title=None,
                         value_format="{:.1f}",
                         bar_width=0.15,
                         palette=PASTEL_PALETTE,
                         ann_thresh=0.5):
    """Grouped bars with restrained annotations."""
    n = len(categories)
    m = len(series_dict)
    group_spacing = 0.3
    x = np.arange(n) * (1 + group_spacing)

    bar_spacing = 0.05
    total_w = m * (bar_width + bar_spacing)
    start = -total_w / 2 + (bar_width + bar_spacing) / 2

    for i, (name, vals) in enumerate(series_dict.items()):
        offs = x + start + i * (bar_width + bar_spacing)
        bars = ax.bar(offs, vals, width=bar_width,
                      label=name,
                      color=palette[i % len(palette)],
                      edgecolor="none")
        for r in bars:
            h = r.get_height()
            ax.annotate(value_format.format(h),
                        (r.get_x() + r.get_width() / 2, h),
                        textcoords="offset points",
                        xytext=(0, 4),
                        ha="center", va="bottom",
                        fontsize=25, weight="bold",
                        clip_on=False)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0, ha="right", weight="bold", fontsize=30,)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=30, weight="bold")
    if title:
        ax.set_title(title, pad=15, fontsize=30,
                     weight="bold", y=1.06, fontstyle="italic")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle=(0, (3, 3)),
            linewidth=1, color="gray", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="both", labelsize=22)

def global_ylim(*series_dicts):
    vals = []
    for sd in series_dicts:
        for v in sd.values():
            vals += [x for x in v if x is not None]
    ymax = max(vals) if vals else 1.0
    return 0, int(np.ceil(ymax / 10.0)) * 10

# Scatterplot: SEP Score (%) vs Utility (%)
# Note: Replace the example data values in `data_points` with your real results.
# This uses matplotlib (no seaborn, no custom colors) and produces a single chart.
# The plot is saved to /mnt/data/sep_vs_utility_scatter.png

# --- Data ---
# categories = ["Undefended", "StruQ", "SecAlign", "ISE", "PFT", "Ours"]
#
# series_l8b_strict = {
#     "Naive":      [5.74, 5.26, 0.00, 0.96, 0.00, 0.00],
#     "Ignore":     [11.00, 27.27, 0.00, 6.70, 8.13, 0.00],
#     "Completion": [0.00, 0.00, 0.00, 0.96, 21.05, 0.00],
#     "Escape":     [6.22, 7.66, 0.00, 1.44, 1.44, 0.00],
#     "HackAPrompt": [23.81, 52.38, 0.00, 0.00, 52.38, 0.00],
#     "GCG":         [98.08, 98.08, 66.67, 98.56, 98.08, 1.06],
# }
#
# series_m7b_strict = {
#     "Naive":      [2.39, 1.44, 0.00, 1.44, 0, 0],
#     "Ignore":     [22.01, 1.91, 1.44, 9.57, 2.39, 0],
#     "Completion": [23.44, 0.00, 0.48, 0.00, 0.00, 0],
#     "Escape":     [11.96, 2.87, 0.48, 3.83, 2.39, 0],
#     "HackAPrompt": [38.10, 47.62, 0.00, 42.86, 19.05, 0],
#     "GCG":         [100.00, 100.00, 98.56, 66.83, 66.83, 3.37],
# }
#
# # --- Create 2x1 subplots ---
# fig, axes = plt.subplots(2, 1, figsize=(40, 12), constrained_layout=False)
#
# plot_grouped_bars_ax(axes[0], categories, series_l8b_strict,
#                      ylabel="ASR (%)",
#                      title="LLaMA-8B")
#
# plot_grouped_bars_ax(axes[1], categories, series_m7b_strict,
#                      ylabel="ASR (%)",
#                      title="Mistral-7B")
#
# # --- Uniform y-axis across both plots ---
# y0, y1 = global_ylim(series_l8b_strict, series_m7b_strict)
# for ax in axes:
#     ax.set_ylim(y0, y1)
#
# # --- Shared legend at top ---
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels,
#            ncols=2,
#            frameon=True,
#            loc="upper right",
#            prop={"weight": "bold", "size": 25})
#
# # --- Global X label ---
# fig.text(0.5, 0.03, 'Defense Method', ha='center',
#          fontsize=30, weight='bold')
#
#
# # --- Adjust spacing ---
# fig.subplots_adjust(top=0.92, bottom=0.08, left=0.04, right=0.96, hspace=0.2)
#
# plt.show()

groups = [
    "Ours",
    "+ Reminder",
    "+ Sandwich",
    "+ Fakecompletion",
    "+ ThinkIntervene",
    "+ SpotlightDelimit",
    "+ SpotlightDatamark",
    "+ SpotlightEncode",
]

data_points = {
    "Ours": (83.5, 16.78),
    "+ Reminder": (82.1, 14.3),
    "+ Sandwich": (79.0, 14.6),
    "+ Fakecompletion": (77.6, 14.6),
    "+ ThinkIntervene": (80.7, 13.2),
    "+ SpotlightDelimit": (81.4, 10.0),
    "+ SpotlightDatamark": (79.0, 6.0),
    "+ SpotlightEncode": (85.2, 8.5),
}

colors = [
    "#1b9e77",  # teal
    "#d95f02",  # orange
    "#7570b3",  # indigo
    "#e7298a",  # magenta
    "#66a61e",  # olive
    "#e6ab02",  # mustard
    "#a6761d",  # brown
    "#666666",  # grey
]

fig, ax = plt.subplots(figsize=(5.5, 4.0))

markers = ["o", "s", "^", "D", "P", "X", "8", "v"]

for i, g in enumerate(groups):
    x, y = data_points[g]
    ax.scatter(
        x, y,
        marker=markers[i % len(markers)],
        s=70,
        linewidths=1.0,
        edgecolors="black",
        facecolors=colors[i % len(colors)],
        label=g
    )
    # dx, dy = label_offsets[g]
    # ax.annotate(
    #     g, (x, y+1),
    #     textcoords="offset points",
    #     xytext=(dx, dy),
    #     # ha="left",
    #     # va="center",
    #     fontsize=11
    # )

ax.set_xlabel("SEP score (%)")
ax.set_ylabel("Utility (%)")

ax.set_xlim(75, 100)
ax.set_ylim(0, 20)

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(2.5))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(2.5))

ax.xaxis.set_major_formatter(PercentFormatter(100))
ax.yaxis.set_major_formatter(PercentFormatter(100))

# light, unobtrusive grid
ax.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.4)
ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.25)

# diagonal reference line within current limits
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
low, high = max(xmin, ymin), min(xmax, ymax)
ax.plot(
    [low, high],
    [low, high],
    linestyle="--",
    linewidth=1.0,
    color="#4d4d4d",  # nicer neutral grey
)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

plt.legend(loc="lower right", frameon=True, prop={"weight": "bold"})
fig.tight_layout()
plt.show()
