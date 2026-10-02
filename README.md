# Resource allocation for multi-objective Bayesian optimization

A Bayesian optimization campaign has a fixed budget. Every batch spends some of
it on exploring the design space and some on exploiting what the surrogate model
already believes. This repository asks whether a large language model can make
that split better than a fixed schedule, and measures the answer on two alloy
design problems against non-adaptive baselines.

Each BO iteration selects a batch of 5 candidate alloys. The split between
qUCB (exploration) and qEHVI (exploitation) within that batch is the decision
under study. Two of the four strategies hand that decision to an LLM; two use a
fixed rule.

| Strategy | Decides the split by | Implementation |
|---|---|---|
| `Adaptive-Beta` | LLM picks a continuous UCB `beta`, then picks an allocation | `adaptive_scout_policy.py` |
| `Rule-Swap` | LLM assesses progress, then picks a discrete exploration level | `qehvi_first_policy.py` |
| `Fixed-Mixed` | Fixed phase schedule driven by budget consumed | `fixed_policy.py` |
| `qEHVI` | Pure exploitation, no exploration | `fixed_policy.py` |

Two alloy systems, each with four experiment variants: a baseline run and three
that inject a mid-campaign disruption (time cut, budget cut, or both), to test
whether the adaptive strategies respond to a change in circumstances.

- **MP-D** — Ti-V-Nb-Mo-Hf-Ta-W refractory alloys. Maximize melting point,
  minimize density. 7 components, 55 iterations.
- **Magnets** — Fe-Co-Ni-V-Mo-Cr-Cu-Mn-C-W-Ta-Nb-Al-Ti-Si. 15 declared
  components, 50 iterations.

Objectives are evaluated against random-forest surrogate models trained on
CALPHAD and experimental data, which stand in for the physical experiment. The
trained models ship in `ground_truth_models/`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Add `[surrogate]` instead of `[dev]` only if you intend to run
`ground_truth.ipynb` (Figure 2) or retrain the ground-truth models; that extra
pulls in TensorFlow, XGBoost, and Optuna, which the optimization loop itself
does not use.

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. The two LLM strategies
call the model once per BO iteration; `Fixed-Mixed` and `qEHVI` run without a
key. Set `OPENAI_BASE_URL` to point at an OpenAI-compatible endpoint other than
OpenAI's.

## Reproduce the experiments

```bash
./replicate.sh
```

That runs all 8 experiments: 25 seeds x 4 strategies each, writing to
`results_magnets_v3/` and `results_mpd_v3/` under
`<experiment>/seed_<N>/<strategy>/`.

Run a slice instead:

```bash
./replicate.sh --dry-run                      # print the plan, run nothing
./replicate.sh --only Magnets-MAIN            # one experiment
./replicate.sh --only MP-D_MAIN --seeds 2024  # one seed
```

Completed seed/strategy pairs are skipped, so an interrupted campaign restarts
with the same command.

**On reproducibility.** `Fixed-Mixed` and `qEHVI` are deterministic given their
seeds. `Adaptive-Beta` and `Rule-Swap` are not: they depend on live LLM
responses, so a rerun will land near the published numbers rather than on them.
The full campaign takes days of wall-clock time and real API spend. Use `--only`
and `--seeds` to run a slice first.

## Reproduce the figures and tables

The published campaign results ship in `campaign_results/` (8 experiments x 25
seeds x 4 strategies, per-iteration convergence, evaluations, candidate options,
and LLM agent logs). From them:

```bash
python analysis/paper_figures.py --out-dir paper/figures   # Figs. 3-8 and all tables
```

Figure 2 and the ground-truth design-space files come from `ground_truth.ipynb`
(needs the `[surrogate]` extra). To plot your own rerun from `replicate.sh`
instead, add `--results-dir .`.

## Data

Both ground-truth models are trained on published datasets, shipped in
`ground_truth_models/data/`. No rows are removed from either.

**Fe-Co-Ni soft magnets, `FeNiCo_comp-prop_imp.csv`** (1,208 rows)

- Source: Padhy et al., "Experimentally validated inverse design of
  multi-property Fe-Co-Ni alloys", *iScience* 27, 109723 (2024). Data and code:
  <https://github.com/Shakti-95/Data-and-Codes-for-Experimentally-Validated-Inverse-design-of-Multi-Property-Fe-Co-Ni-alloys>,
  archived at [doi:10.5281/zenodo.10686272](https://doi.org/10.5281/zenodo.10686272).
- This is `2-Imputation/FeNiCo_comp-prop_imp.csv` from that repository (commit
  `0ab1b99`), a literature-compiled database whose missing property values were
  filled by Padhy et al. with per-property ML models (see their `2-Imputation/`).
  The only additions here are log10 columns: `LogHC`, `LogTC` (from Tc converted
  from °C to K), `LogER`, `LogElong`, `LogHV`.
- Share of values that are measured rather than imputed: Ms 75%, Hc 50%, Tc 72%,
  electrical resistivity 35%, elongation 20%, HV 14%.
- Inputs: the 15 element columns in at.%, divided by 100 to give atomic
  fractions. Outputs: `Ms`, `LogHC`, `LogTC`, `LogER`, `LogElong`, `LogHV`; the
  campaigns optimize Ms, log Hc and log HV.

**Ti-V-Nb-Mo-Hf-Ta-W refractory alloys, `refractory-meltingvsdensity.xlsx`**
(1,000 rows)

- Source: Hastings et al., "Leveraging domain knowledge for optimal
  initialization in Bayesian materials optimization", *Digital Discovery* (2025).
  Computed, not measured, properties of 1,000 distinct compositions.
- Inputs: the 7 element columns, already atomic fractions. Outputs:
  `Melting Point (K)` and `PROP RT Density (g/cm3)`. The remaining columns are
  unused.

`ground_truth.ipynb` evaluates both models on an 80/20 random split
(`random_state=42`) and generates each system's enumerated design space
(`design_space_inputs.csv`, `design_space_predictions.csv`) from the settings in
`src/resource_allocation/experiments/config.py`.

## Reported numbers

- Hypervolume is normalized by the oracle hypervolume of the full enumerated
  design space: **444.219358** for MP-D, **0.968626** for Magnets.
- n = 25 seeds per experiment per strategy.
- Strategy comparisons use the paired Wilcoxon signed-rank test, paired by
  shared random seed.
- Pre-event versus baseline comparisons use Mann-Whitney U, unpaired, because
  the seed sets differ.
- Error bands in figures are 95% confidence intervals on the mean across seeds,
  t-distribution critical value (n - 1 degrees of freedom) x SEM.

## Layout

```
src/resource_allocation/   the library, plus experiments/ (runner and configs)
analysis/                  figure and statistics scripts
tests/                     pytest suite, no GPU or API key needed
ground_truth_models/       surrogate models, design spaces, raw training data
campaign_results/          published campaign results behind the paper
paper/                     LaTeX manuscript
figures/                   generated figure output
replicate.sh               runs the experiment campaign
```

`AGENT.md` is the deeper orientation: module dependency order, what each
component owns, where the LLM prompts live, and the schema of the result files.

## Tests

```bash
pytest
```

28 tests covering the GP wrapper, acquisition functions, the decision state
contract, logging, visualization, and a full BO loop on synthetic objectives.
No GPU, no trained models, no API calls.

## Citing

See `CITATION.cff`.

## License

MIT, see `LICENSE`.
