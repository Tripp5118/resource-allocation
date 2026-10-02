# Orientation

For an agent picking up this repository cold. Read `README.md` first for what the
project studies; this file covers how the code is arranged and in what order
things happen.

## What runs

One entry point: `src/resource_allocation/experiments/run_campaign.py`. It runs
one experiment, meaning every seed in that experiment's hardcoded list crossed
with all four strategies. `replicate.sh` loops it over all eight experiments.

```bash
python -m resource_allocation.experiments.run_campaign \
    --experiment Magnets-MAIN --output-dir results_magnets_v3
```

`--experiment` is required and must be one of the eight keys in
`experiments/config.py`. `--output-dir` is required. `--seeds` restricts to
specific seeds, which must already be in that experiment's list. `--dry-run`
prints the plan without executing.

All paths in `experiments/config.py` are relative to the repository root, so run
from there. `replicate.sh` changes directory for you.

## Module dependency order

Four layers, acyclic. Nothing in a lower layer imports from a higher one.

```
leaves         decision_state  design_space  gp_models  acq_ucb
               truth_interface  logging_utils  armote_multi_output

mid            decision_maker  -> decision_state
               visualization   -> logging_utils

policies       fixed_policy  qehvi_first_policy  adaptive_scout_policy
               llm_decision_maker  agent_manager
                              -> decision_maker + decision_state

orchestrator   batched_runs    -> design_space, truth_interface, gp_models,
                                  acq_ucb, agent_manager, fixed_policy,
                                  logging_utils, visualization
```

`batched_runs.run_bo_experiment()` is the loop. It is the only place the whole
stack comes together.

## What each component owns

**`decision_state.py`** — the data contract between the loop and any policy:
`DecisionState`, `StrategyOutcome`, `ParetoStatus`, `ProgressVelocity`,
`StrategyEffectiveness`. Change this and every policy is affected.

**`design_space.py`** — `DesignSpace` enumerates the discretized composition
simplex under per-element bounds, with optional group-level sum constraints
(majors versus dopants in the Magnets system). MP-D uses step 0.05, Magnets
step 0.02.

**`truth_interface.py`** — `TruthModelEvaluator` wraps a pickled random-forest
surrogate plus its x and y scalers. This is what stands in for running a physical
experiment. Unpickling is sensitive to the scikit-learn version; the pin in
`pyproject.toml` is load-bearing.

**`gp_models.py`** — `GPModelManager` fits a BoTorch `MultiTaskGP` and predicts
in chunks. Defines the module-level `DEVICE` and `DTYPE`.

**`acq_ucb.py`** — the acquisition engine. Optimizes qEHVI and qUCB over the
discrete simplex, computes the mutual-information term, and builds the five
`AllocationOption` candidates a policy chooses between. Imports nothing else from
this package and declares its own `DEVICE`/`DTYPE`, duplicating `gp_models`.

**`agent_manager.py`** — `BOAgent` is the wrapper the loop is handed. `ResourceEvent`
and `EventManager` apply the mid-campaign time and budget disruptions.

**`batched_runs.py`** — the BO loop, plus hypervolume computation, shared
initialization across strategies, and normalization. Carries a second event
manager, `SimpleResourceEventManager`, parallel to the one in `agent_manager`.

**`logging_utils.py`** — `LoggingManager` owns every output file (schema below).

**`visualization.py`** — Pareto and hypervolume plots. Disabled during campaign
runs (`create_vis=False`).

**`armote_multi_output.py`** — the Optuna/TPE workflow that trained the
ground-truth surrogates. Not imported by the optimization loop. It is the only
reason the `surrogate` extra exists.

**`llm_decision_maker.py`** — a retired three-stage reflection/belief/decision
pipeline. `run_campaign.py` does not use it; `tests/test_integration.py` still
exercises it. Kept for reference.

## The four strategies

`STRATEGY_CONFIGS` in `run_campaign.py` fixes the set. The `name` is also the
results directory name.

| name | type | class |
|---|---|---|
| `Rule-Swap` | `qehvi_first` | `QEHVIFirstDecisionMaker` |
| `Adaptive-Beta` | `adaptive_scout` | `AdaptiveScoutDecisionMaker` |
| `Fixed-Mixed` | `fixed_mixed` | `FixedMixedPolicy(beta=2.0)` |
| `qEHVI` | `qEHVI` | `FixedExploitPolicy(beta=2.0)` |

`build_strategy()` in `run_campaign.py` constructs them. The two LLM strategies
each get a `ChatOpenAI` client there.

## What the LLM actually decides

Two sequential calls per BO iteration, in both LLM strategies.

**Call 1 — `select_beta()`.** The model sees campaign state (iteration, fraction
of resources consumed, time remaining, Pareto hypervolume and its delta, Pareto
set size), raw optimization signals (`velocity_ratio`, `mi_current`,
`mi_vs_start`), per-element composition coverage of the current Pareto set, and a
rolling history of roughly the last six iterations. It returns a UCB `beta`.
Python then computes the five allocation options at that beta.

**Call 2 — `make_decision()`.** The model sees its own beta and reasoning from
call 1, plus the options table with each option's actual qEHVI and mutual
information scores. It returns an option index, which fixes how many of the five
batch points go to qUCB versus qEHVI.

Python enforces nothing about the reasoning. It parses the output and bounds-checks
it; on a parse failure beta falls back to 2.0. In `AdaptiveScoutDecisionMaker` the
two calls are `_call1` and `_call2`. In `QEHVIFirstDecisionMaker` they are
`_call1_assessment_and_beta` and `_call2_option_selection`, where call 1 is a
progress assessment rather than a free choice of beta, and option 0 (all qEHVI) is
the stated default.

**Prompts are inline f-strings inside those methods**, not template files and not
config. The module-level constants `_ACQ_FRAMING`, `_BETA_GUIDANCE` (adaptive
scout) and `_ACQ_FRAMING`, `_URGENCY_FRAMING` (qehvi first) hold the shared
preamble text. The only prompt text in config is `problem_description` per
experiment in `experiments/config.py`.

Editing a prompt changes the method under study. The manuscript reproduces these
prompts by hand, so the two can drift.

## Result file schema

```
<output-dir>/<experiment>/seed_<N>/<strategy>/
    <strategy>_convergence.csv    one row per iteration, 46 columns:
                                  iteration, total_hypervolume, information_gain,
                                  selected_n_exp, selected_n_opt, time_remaining,
                                  budget_remaining, pareto_size, best_x_0..best_x_14
    <strategy>_options.csv        the five allocation options scored each iteration
    <strategy>_evaluations.csv    every candidate evaluated
    <strategy>_metadata.json      run configuration
    agent_logs/                   LLM strategies only
        adaptive_scout_iteration_<N>.json   key "beta"
        qehvi_first_iteration_<N>.json      key "call1_beta"
<output-dir>/<experiment>/failed_strategies.log
<output-dir>/_run_info.txt
```

**Beta is not in the convergence CSV.** It exists only in the `agent_logs/` JSON
files, so those cannot be pruned if anything downstream plots beta.

`is_strategy_complete()` in `run_campaign.py` is the skip-if-done guard. It counts
non-blank lines in the convergence CSV and requires at least `iters + 1`.

## Ground truth

`ground_truth_models/` holds, per system, a `models/` directory (random forest
plus x and y scalers) and two enumerated design-space CSVs. `data/` holds the raw
CALPHAD and experimental spreadsheets the surrogates were trained on.

`ground_truth.ipynb` covers both surrogates and needs the `[surrogate]` extra. Run
from the repository root, it writes Figure 2 (`figures/fig2_parity_plots.png`),
per-system ARMOTE parity plots in `figures/parity/`, and both systems' design-space
CSVs. With the default `RETRAIN = False` it evaluates the committed models on
ARMOTE-MO's 80/20 split (`random_state=42`), which reproduces the published $R^2$
values and the committed CSVs byte for byte. `RETRAIN = True` reruns
`armote_multi_output.run_workflow()` and overwrites the pickles. Optuna search is
not seeded, so retraining will not reproduce either pickle exactly, and every
published number is pinned to the committed ones.

## Figures and tables

`analysis/paper_figures.py` generates every data-driven figure (Figs. 3-8) and
table in the paper:

```bash
python analysis/paper_figures.py --out-dir paper/figures              # everything
python analysis/paper_figures.py --out-dir paper/figures fig5 tables
```

`--results-dir` is the directory holding `results_mpd_v3/` and
`results_magnets_v3/`. It defaults to `campaign_results/`, the published runs
behind every number in the paper. `replicate.sh` writes new runs to the
repository root instead, so a rerun never mixes with the published data; pass
`--results-dir .` to plot a rerun. Function names match paper figure numbers
(`make_fig5_mi_tradeoff()` writes `fig5_mi_tradeoff.png`). Besides the PNGs it
writes `tab_cumulative_mi_full.tex` and `tab_significance.tex` (input by the
appendix), `tab_convergence_speed.tex` (the body of Table
`tab:convergence-speed`, which is currently inlined in
`03_results_discussion.tex`), `stats_table.txt`/`stats_table.tex`, and plain-text
summaries for Fig. 5(c) and the milestone table. `make_fig5_mi_tradeoff()` warns,
rather than failing, if regenerated values drift from the published ones.

- `compute_gap_stats.py` and `export_paper_data.py` also read
  `campaign_results/`.
- Figure 2 (`fig2_parity_plots.png`) comes from `ground_truth.ipynb`, not from
  this script.

## Tests

```bash
pytest
```

28 tests, no GPU, no trained models, no API calls. `tests/conftest.py` provides
synthetic training data and a `mock_llm` fixture returning a canned response.
`tests/test_integration.py` forces CPU by overwriting the `DEVICE` constant in
three modules before anything else imports them, so leave those assignments at
the top of the file.

## Conventions worth keeping

- The package is importable as `resource_allocation` after `pip install -e .`.
  There is no `sys.path` manipulation in the library; do not add any.
- Experiment parameters live in `experiments/config.py`, not in the runner.
- Anything that changes how candidates are selected, scored, or evaluated is a
  change to the method the paper reports, not a refactor. That covers
  `gp_models.py`, `acq_ucb.py`, `batched_runs.py`, `design_space.py`, the policy
  modules, and the prompts inside them.
