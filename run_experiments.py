"""
runs every experiment in parallel on all cpu cores but one
results get collected in memory and written once at the end which replaces the old csv files
so results/ only has the seeds in SEEDS and if you stop it halfway nothing gets saved
results/run_log.csv never gets replaced and every finished run just adds a line with how long it took

each method game and seed is its own job with its own random seed
so running them in parallel gives the exact same numbers as running them one by one
and the same seed gives the same numbers every time as long as its the same pc code and package versions

seeds budget and games are set in src/common.py btw
"""
import os
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count

import pandas as pd

from src.common import *
from src.baselines import random_policy_baseline, rule_based_baseline, random_search
from src.dfo import dfo, VARIANTS

# each group writes <group>_summary.csv and <group>_curves.csv
GROUPS = ["baselines", "dfo", "ext"]


def make_jobs():
    jobs = []
    for seed in SEEDS:
        for env_name in ENVS:
            for method in ["random_policy", "rule_based", "rule_based_flipped", "random_search"]:
                jobs.append(("baselines", method, env_name, seed))
            jobs.append(("dfo", "dfo", env_name, seed))
        for method in VARIANTS:                          # extension variants: Acrobot only
            if method != "dfo":
                jobs.append(("ext", method, "Acrobot-v1", seed))
    return jobs


def run_job(job):
    # runs one experiment and returns its group summary row curve rows and how many seconds it took
    # it has to stay a top level function or windows multiprocessing stops working
    group, method, env_name, seed = job
    start_time = time.time()
    curve = []
    if method == "random_policy":
        mean_return, std_return = random_policy_baseline(env_name, seed)
    elif method in ("rule_based", "rule_based_flipped"):
        mean_return, std_return = rule_based_baseline(env_name, seed, flipped=(method == "rule_based_flipped"))
    elif method == "random_search":
        summary, curve = random_search(env_name, seed)
    else:
        summary, curve = dfo(env_name, seed, method_name=method, **VARIANTS[method])
    if method in ("random_policy", "rule_based", "rule_based_flipped"):
        summary = {"method": method, "env": env_name, "seed": seed,
                   "test_mean": mean_return, "test_std": std_return, "solved": mean_return >= THRESHOLDS[env_name]}
    return group, summary, curve, time.time() - start_time


def log_run(n_jobs, workers, minutes):
    # adds one line to results/run_log.csv with this runs timing and is never overwritten
    row = {"finished": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "n_seeds": len(SEEDS), "seeds": f"{min(SEEDS)}-{max(SEEDS)}",
           "n_runs": n_jobs, "workers": workers,
           "total_minutes": round(minutes, 1),
           "seconds_per_run": round(minutes * 60 / n_jobs, 1)}
    log_path = os.path.join(RESULTS_DIR, "run_log.csv")
    pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)


def main():
    jobs = make_jobs()

    workers = max(1, min(cpu_count() - 1, len(jobs)))
    print(f"Running {len(jobs)} jobs ({len(SEEDS)} seeds) on {workers} worker(s)")
    t_start = time.time()
    summaries = {group: [] for group in GROUPS}
    curves = {group: [] for group in GROUPS}
    with Pool(workers) as pool:
        for job_number, (group, summary, curve, secs) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            summaries[group].append(summary)
            curves[group] += curve
            print(f"[{job_number}/{len(jobs)}] {summary['method']:<20} {summary['env']:<12} seed {summary['seed']}: "
                  f"test {summary['test_mean']:7.1f}  ({secs:.0f}s)")

    # write each file once; to_csv overwrites the old file
    for group in GROUPS:
        pd.DataFrame(summaries[group]).to_csv(os.path.join(RESULTS_DIR, f"{group}_summary.csv"), index=False)
        pd.DataFrame(curves[group]).to_csv(os.path.join(RESULTS_DIR, f"{group}_curves.csv"), index=False)

    minutes = (time.time() - t_start) / 60
    log_run(len(jobs), workers, minutes)
    print(f"\nDone in {minutes:.1f} min (logged in results/run_log.csv). Run analyse_results.py next.")


if __name__ == "__main__":      # required on Windows: worker processes re-import this file
    main()
