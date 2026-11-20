# makes the charts and the results table and only reads results so it never runs experiments
import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common import *

warnings.filterwarnings("ignore", message="All-NaN slice")   # grid points before a seed's first evaluation

LABELS = {"random_policy": "Random policy", "rule_based_flipped": "Rule (wrong sign)",
          "rule_based": "Rule-based", "random_search": "Random search",
          "dfo": "DFO", "dfo_adaptive": "DFO + adaptive dt", "dfo_reeval": "DFO + re-evaluation",
          "dfo_adaptive_reeval": "DFO + both"
          }
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
MARK = "#2a78d6"

# figure sizes: screen, or one report column
SIZES = {"screen": dict(w=5.2, h_sum=3.2, h_cur=3.4, title=11, small=8, base=10, ms=8),
         "column": dict(w=3.4, h_sum=2.5, h_cur=2.3, title=8, small=6.5, base=7, ms=5)}

# fixed colour and line style per method
STYLE = {"random_search":       ("#2a78d6", "-"),     # blue, solid
         "dfo":                 ("#eb6834", "--"),    # orange, dashed
         "dfo_adaptive_reeval": ("#1baf7a", ":")}     # aqua, dotted

# games whose curves start far below the rest so the y axis gets zoomed in on the later part
ZOOM_ENVS = ["Acrobot-v1"]
ZOOM_SKIP = 0.1    # part of the budget left out when picking the zoomed y range


def load_results(results_dir=RESULTS_DIR):
    # loads every saved summary and curve file and joins them into one table each
    summary_files = sorted(glob.glob(os.path.join(results_dir, "*_summary.csv")))
    curve_files = sorted(glob.glob(os.path.join(results_dir, "*_curves.csv")))
    if not summary_files:
        raise FileNotFoundError(f"No *_summary.csv files in {results_dir}. Run run_experiments.py first.")
    summary_df = pd.concat([pd.read_csv(path) for path in summary_files], ignore_index=True)
    curves_df = pd.concat([pd.read_csv(path) for path in curve_files], ignore_index=True)
    print("Files:", [os.path.basename(path) for path in summary_files])
    print("\nRuns per method:")
    print(summary_df.groupby(["env", "method"]).size().to_string())
    return summary_df, curves_df


def _finish(fig, filename, out_dir=FIGURES_DIR):
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary_df, filename="all_methods_summary.png", size="screen", out_dir=FIGURES_DIR):
    settings = SIZES[size]
    plt.rcParams.update({"font.size": settings["base"]})
    envs = list(summary_df["env"].unique())
    fig, axes = plt.subplots(1, len(envs), figsize=(settings["w"] * len(envs), settings["h_sum"]))
    axes = np.atleast_1d(axes)
    for ax, env_name in zip(axes, envs):
        env_data = summary_df[summary_df["env"] == env_name]
        methods = [method for method in LABELS if method in env_data["method"].unique()] + \
                  [method for method in env_data["method"].unique() if method not in LABELS]
        for row, method in enumerate(methods):
            scores = env_data[env_data["method"] == method]["test_mean"].to_numpy()
            ax.scatter(scores, np.full(len(scores), row) + np.linspace(-0.12, 0.12, len(scores)),
                       s=14 if size == "screen" else 7, color=MARK, alpha=0.35, zorder=2, linewidths=0)          # individual seeds
            ax.errorbar(scores.mean(), row, xerr=scores.std(ddof=1) if len(scores) > 1 else 0, fmt="o", ms=settings["ms"], color=MARK,
                        ecolor=MARK, elinewidth=2, capsize=0, zorder=3)              # mean ± sd (same sd as the table)
            ax.annotate(f"{scores.mean():.1f}", (scores.mean(), row), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=settings["small"], color=INK, zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
        thr = THRESHOLDS[env_name]
        ax.axvline(thr, color=MUTED, lw=1, ls="--", zorder=1)
        ax.text(thr, -0.55, f" solved ({thr:g})", color=MUTED, fontsize=settings["small"], va="bottom")
        ax.set_yticks(range(len(methods)), [LABELS.get(method, method) for method in methods], color=INK)
        ax.set_ylim(-0.6, len(methods) - 0.1)
        ax.set_xlabel(f"Test return ({TEST_EPISODES} fresh episodes)", color=MUTED)
        ax.set_title(env_name, color=INK, fontsize=settings["title"], loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0)
    _finish(fig, filename, out_dir)


def curves_on_grid(curves_df, env_name, method, n_points=200):
    # lines up every seeds best score so far on the same step grid so the seeds can be combined
    grid = np.linspace(0, BUDGET, n_points)
    runs = []
    method_data = curves_df[(curves_df["env"] == env_name) & (curves_df["method"] == method)]
    for seed, seed_data in method_data.groupby("seed"):
        seed_data = seed_data.sort_values("steps")
        steps = seed_data["steps"].to_numpy()
        best = seed_data["best_score"].to_numpy()
        idx = np.searchsorted(steps, grid, side="right") - 1           # last evaluation at/before each grid point
        runs.append(np.where(idx >= 0, best[np.clip(idx, 0, None)], np.nan))
    return grid, np.array(runs)


def label_line_ends(ax, ends, min_gap_frac=0.07, fontsize=8):
    # writes each lines name at its right end and spreads the names out so none of them overlap
    if not ends:
        return
    y_min, y_max = ax.get_ylim()
    gap = (y_max - y_min) * min_gap_frac
    ends = sorted(ends, key=lambda end: end[1])
    label_ys = [end[1] for end in ends]
    for index in range(1, len(label_ys)):                      # push each label up until it clears the one below
        label_ys[index] = max(label_ys[index], label_ys[index - 1] + gap)
    shift = max(0, label_ys[-1] - y_max)                      # if the top one went off the chart, slide all down
    x_right = ax.get_xlim()[1]
    for (label, end_value), label_y in zip(ends, label_ys):
        ax.annotate(label, (x_right, label_y - shift), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=fontsize, color=INK, annotation_clip=False)


def plot_curves(curves_df, filename="all_methods_curves.png", methods=None,
                size="screen", out_dir=FIGURES_DIR):
    settings = SIZES[size]
    plt.rcParams.update({"font.size": settings["base"]})
    envs = list(curves_df["env"].unique())
    fig, axes = plt.subplots(1, len(envs), figsize=(settings["w"] * len(envs), settings["h_cur"]))
    axes = np.atleast_1d(axes)
    for ax, env_name in zip(axes, envs):
        present = curves_df[curves_df["env"] == env_name]["method"].unique()
        methods_here = [method for method in (methods or STYLE) if method in STYLE and method in present]
        ends = []
        lows = []
        highs = []
        seed_counts = []
        for method in methods_here:
            colour, line_style = STYLE[method]
            grid, runs = curves_on_grid(curves_df, env_name, method)
            med = np.nanmedian(runs, axis=0)
            band_low = np.nanmin(runs, axis=0)
            band_high = np.nanmax(runs, axis=0)
            ax.fill_between(grid, band_low, band_high, color=colour, alpha=0.15, linewidth=0)
            later = grid >= BUDGET * ZOOM_SKIP
            lows.append(np.nanmin(band_low[later]))
            highs.append(np.nanmax(band_high[later]))
            seed_counts.append(len(runs))
            ax.plot(grid, med, color=colour, ls=line_style, lw=2 if size == "screen" else 1.5, label=LABELS.get(method, method))
            ends.append((LABELS.get(method, method), med[-1]))
        thr = THRESHOLDS[env_name]
        if env_name in ZOOM_ENVS and lows:                     # zoom on the part after the first few rounds
            y_low = min(min(lows), thr)
            y_high = max(max(highs), thr)
            padding = (y_high - y_low) * 0.08
            ax.set_ylim(y_low - padding, y_high + padding)
        ax.axhline(thr, color=MUTED, lw=1, ls="--")
        ax.text(BUDGET * 0.5, thr, f"solved ({thr:g})", color=MUTED, fontsize=settings["small"], va="top", ha="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
        ax.set_title(env_name, color=INK, fontsize=settings["title"], loc="left")
        ax.set_xlabel("Environment steps", color=MUTED)
        ax.set_ylabel("Best training score so far", color=MUTED)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, position: f"{value/1000:.0f}k"))
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0)
        if size == "screen":                                    # column figures are too narrow for end labels
            label_line_ends(ax, ends, fontsize=settings["small"])      # after plotting, so the y-limits are final
        if len(methods_here) >= 2:                              # a legend only once there are 2+ lines
            # seed count comes from the data so the legend stays right for any number of seeds
            if min(seed_counts) == max(seed_counts):
                seeds_text = f"{seed_counts[0]}"
            else:
                seeds_text = f"{min(seed_counts)}-{max(seed_counts)}"
            ax.legend(frameon=False, fontsize=settings["small"], loc="lower right",
                      title=f"median of {seeds_text} seeds, band = min to max", title_fontsize=settings["small"])
    _finish(fig, filename, out_dir)


def results_table(summary_df, results_dir=RESULTS_DIR):
    # builds the results table with one row for each game and method and saves it to results/results_table.csv
    table = summary_df.groupby(["env", "method"]).agg(
            test_mean=("test_mean", "mean"), test_std=("test_mean", "std"),
            solved_pct=("solved", "mean"),
            median_steps_to_threshold=("steps_to_threshold", "median"),
            best_steps_to_threshold=("steps_to_threshold", "min"))      # Wong et al. report the best of 5
    table["solved_pct"] = table["solved_pct"] * 100
    table = table.round(1)
    table.to_csv(os.path.join(results_dir, "results_table.csv"))
    return table

def optimism_gaps(summary_df, curves_df):
    # for each run the final best training score minus the test score
    # a large gap means the optimiser was fooled by lucky noisy games which is called the winners curse
    # only random search and the dfo versions have training curves so only they get a gap
    train = curves_df.sort_values("steps").groupby(["env", "method", "seed"])["best_score"].last()
    test = summary_df.set_index(["env", "method", "seed"])["test_mean"]
    gap = (train - test.reindex(train.index)).rename("gap").reset_index()
    gap["train_best"] = train.values
    gap["test_mean"] = test.reindex(train.index).values
    return gap


def gap_table(gap_df, results_dir=RESULTS_DIR):
    # average training best test score and optimism gap for each method saved to results/optimism_gap.csv
    table = gap_df.groupby(["env", "method"]).agg(
            train_best=("train_best", "mean"), test_mean=("test_mean", "mean"),
            gap_mean=("gap", "mean"), gap_median=("gap", "median"), gap_max=("gap", "max")).round(1)
    table.to_csv(os.path.join(results_dir, "optimism_gap.csv"))
    return table
