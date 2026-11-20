"""
reads the csvs in results/ and makes the charts the results table and the stats
it never reruns an experiment so it only takes a few seconds

what it makes
  figures/*.png                 charts for looking at on screen
  figures/acrobot_*.pdf         charts sized for one column of the report
  results/results_table.csv     average test score spread percent solved and steps to threshold per method
  results/statistics.csv        test score comparisons
  results/statistics_gap.csv    optimism gap comparisons

how the stats work
the main method gets compared with every other method seed by seed
thats dfo + both on acrobot and plain dfo on cartpole
the random policy and wrong sign rule get left out since theyre just there as a floor
  mann whitney u test checks if the two sets of seed scores could come from the same place
  holm correction makes the p values stricter since we do lots of tests at once
  a12 effect size is the chance a random seed from the main method beats a random seed from the other one
    0.5 means no difference and 1.0 means it always wins and around 0.71 or more counts as a big effect
  seeds solved is shown too since reliability is kind of the whole point
statistics.csv compares test scores and statistics_gap.csv compares the optimism gap
the gap is final training best minus test score so basically how much a method got fooled by lucky games
"""
import os
from math import comb

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.common import *
from src.plotting import load_results, plot_summary, plot_curves, results_table, LABELS, optimism_gaps

REFERENCE = {"Acrobot-v1": "dfo_adaptive_reeval", "CartPole-v1": "dfo"}
# baselines left out of the tests
SKIP = {"random_policy", "rule_based_flipped"}


def a12(scores_a, scores_b):
    # effect size that gives the chance a score from group a beats a score from group b where 0.5 means no difference
    wins = 0.0
    for score_a in scores_a:
        for score_b in scores_b:
            if score_a > score_b:
                wins += 1
            elif score_a == score_b:
                wins += 0.5
    return wins / (len(scores_a) * len(scores_b))


def holm(pvals):
    # holm correction that makes each p value stricter because many tests are run at once and returns them in the same order
    p_values = np.asarray(pvals, dtype=float)
    order = np.argsort(p_values)
    adj = np.empty_like(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adj[index] = min(1.0, running)
    return adj


def compare(data, value, higher_is_better=True):
    # compares the main method with every other method seed by seed using one column of results and skips the random policy and wrong sign rule
    rows = []
    for env_name, ref in REFERENCE.items():
        env_data = data[data["env"] == env_name]
        if ref not in env_data["method"].unique():
            continue
        ref_scores = env_data[env_data["method"] == ref][value].to_numpy()
        env_rows = []
        for method in env_data["method"].unique():
            if method == ref or method in SKIP:
                continue
            other = env_data[env_data["method"] == method][value].to_numpy()
            if np.ptp(np.concatenate([ref_scores, other])) == 0:
                p_value = 1.0                                      # all scores identical -> no difference
            else:
                p_value = mannwhitneyu(ref_scores, other, alternative="two-sided").pvalue
            env_rows.append({"env": env_name, "reference": LABELS.get(ref, ref),
                             "compared_with": LABELS.get(method, method),
                             "ref_mean": ref_scores.mean(), "other_mean": other.mean(),
                             "ref_solved": f"{int(env_data[env_data['method'] == ref]['solved'].sum())}/{len(ref_scores)}",
                             "other_solved": f"{int(env_data[env_data['method'] == method]['solved'].sum())}/{len(other)}",
                             "A12": a12(ref_scores, other) if higher_is_better else a12(other, ref_scores),
                             "p": p_value})
        if env_rows:
            for row, p_adj in zip(env_rows, holm([row["p"] for row in env_rows])):
                row["p_holm"] = p_adj
        rows += env_rows
    return rows


def show(rows, path, title):
    out = pd.DataFrame(rows).round({"ref_mean": 1, "other_mean": 1, "A12": 2, "p": 4, "p_holm": 4})
    out.to_csv(path, index=False)
    print(f"\n{title}")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(out.to_string(index=False))
    print(f"Saved to results/{os.path.basename(path)}")


def make_figures(summary_df, curves_df):
    plot_summary(summary_df, "all_methods_summary.png")
    plot_curves(curves_df, "all_methods_curves.png")

    # Extension ablation on Acrobot only (the main result of the report)
    ablation = ["dfo", "dfo_adaptive", "dfo_reeval", "dfo_adaptive_reeval", "random_search", "rule_based"]
    acro = summary_df[(summary_df["env"] == "Acrobot-v1") & (summary_df["method"].isin(ablation))]
    acro_curves = curves_df[curves_df["env"] == "Acrobot-v1"]
    plot_summary(acro, "acrobot_ablation_summary.png")

    # One-column PDF versions for the report
    plot_summary(acro, "acrobot_summary.pdf", size="column", out_dir=FIGURES_DIR)
    plot_curves(acro_curves, "acrobot_curves.pdf", size="column", out_dir=FIGURES_DIR)

    with pd.option_context("display.width", 120, "display.max_columns", 10):
        print("\nResults table (also saved to results/results_table.csv):")
        print(results_table(summary_df))
    print("\nFigures saved to figures/")


def run_statistics(summary_df, curves_df):
    n_seeds = summary_df.groupby(["env", "method"]).size().min()
    print(f"\nSeeds per method: {n_seeds}  (smallest possible two-sided Mann-Whitney p for {n_seeds} vs {n_seeds}: "
          f"{2 / comb(2 * n_seeds, n_seeds):.2g})")

    # Test score (higher is better)
    rows = compare(summary_df, "test_mean")
    show(rows, os.path.join(RESULTS_DIR, "statistics.csv"),
         "Test return (A12 > 0.5 = reference scores higher):")

    # Optimism gap (smaller is better)
    gaps = optimism_gaps(summary_df, curves_df).rename(columns={"gap": "gap_value"})
    gaps["solved"] = gaps["test_mean"] >= gaps["env"].map({"Acrobot-v1": -100.0, "CartPole-v1": 475.0})
    rows = compare(gaps, "gap_value", higher_is_better=False)
    show(rows, os.path.join(RESULTS_DIR, "statistics_gap.csv"),
         "Optimism gap (A12 > 0.5 = reference is fooled LESS):")


def main():
    summary_df, curves_df = load_results()
    make_figures(summary_df, curves_df)
    run_statistics(summary_df, curves_df)


if __name__ == "__main__":
    main()
