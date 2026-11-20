# COMP1818 coursework: DFO for linear-policy reinforcement learning

Dispersive Flies Optimisation (DFO) trains simple linear policies for two Gymnasium games,
CartPole-v1 and Acrobot-v1. It is compared with a random policy, a hand-written rule,
random search, and two changes to DFO (adaptive dt and re-evaluation), tested as an ablation.

## 1. Install

Tested with Python 3.13 on Windows. Install each package (the exact versions used for the results):

    pip install gymnasium==1.4.0
    pip install numpy==2.5.5
    pip install pandas==3.1.1
    pip install scipy==1.18.1
    pip install matplotlib==3.11.2

Other versions should work, but the numbers may differ slightly.

## 2. Run the files in this order

| Step | File | What it does | Time |
|---|---|---|---|
| 1 | `check_setup.py` | Checks the install and tests the policy code. Test 2 should print `(-500.0, 1000)` | seconds |
| 2 | `run_experiments.py` | Runs every experiment and saves the results to `results/` | ~40 min for 21 seeds |
| 3 | `analyse_results.py` | Makes the figures, results table and statistics from `results/` | seconds |

To run a file: open it in PyCharm and click Run, or in a terminal in this folder type `python check_setup.py` (and so on).

`results/` already contains the full 21-seed results, so step 2 is only needed to reproduce them.

## 3. Settings

All settings are in `src/common.py`:
- `SEEDS = list(range(21))`: the seeds used. Set it to e.g. `[0, 1]` for a quick test run
- `BUDGET = 500_000`: environment steps per run
- `EPS_PER_EVAL = 3`: games per evaluation
- `TEST_EPISODES = 20`: games in the final test

The DFO settings (10 flies, dt = 0.001, the extension options) are in `src/dfo.py`.

## 4. What each file does

The scripts you run use the helper files in `src/`:

| File | Details |
|---|---|
| `src/common.py` | used by: everything<br>purpose: settings, folder paths, the observation normaliser, the linear policy, `evaluate` and the final test |
| `src/baselines.py` | used by: `run_experiments.py`<br>purpose: the random policy, the rule-based controller and random search |
| `src/dfo.py` | used by: `run_experiments.py`<br>purpose: DFO, plus the extension options and the variants tested |
| `src/plotting.py` | used by: `analyse_results.py`<br>purpose: loading the results, the figures, and the results and optimism-gap tables |

## 5. What gets made

| Folder | Files |
|---|---|
| `results/` | `baselines_*.csv`, `dfo_*.csv`, `ext_*.csv` (raw results), `results_table.csv`, `statistics.csv`, `statistics_gap.csv`, `run_log.csv` (run times) |
| `figures/` | 3 PNG charts and 2 one-column PDF charts for the report |
| `models/` | The best weights and normaliser for every run (`.npz`) |

## 6. Notes
- Each run of `run_experiments.py` replaces the old results; `run_log.csv` keeps a line for every finished run.
- Same PC, code and package versions give exactly the same numbers. A different PC can give slightly different numbers for some seeds.
