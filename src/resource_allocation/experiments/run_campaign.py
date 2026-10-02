#!/usr/bin/env python3
"""
Runs one experiment: every seed in the hardcoded list, every strategy.

The four strategies are fixed (see STRATEGY_CONFIGS below): two LLM-driven
(Rule-Swap, Adaptive-Beta) and two baselines (Fixed-Mixed, qEHVI). Results go to
<output-dir>/<experiment>/seed_<N>/<strategy>/.

A seed x strategy is skipped if its convergence CSV already holds every expected
row, so an interrupted campaign can be restarted without losing work. CUDA errors
are retried up to MAX_RETRIES times before the strategy is recorded as failed.

To run the whole campaign across all eight experiments, use replicate.sh rather
than calling this module directly.

Usage:
    python -m resource_allocation.experiments.run_campaign \\
        --experiment Magnets-MAIN --output-dir results_magnets_v3
    python -m resource_allocation.experiments.run_campaign \\
        --experiment Magnets-TIME --seeds 2489 9596 --output-dir results_magnets_v3
    python -m resource_allocation.experiments.run_campaign \\
        --experiment MP-D_MAIN --output-dir results_mpd_v3 --dry-run
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from resource_allocation.design_space import DesignSpace
from resource_allocation.batched_runs import (
    run_bo_experiment,
    compute_normalization_params,
    generate_shared_initialization,
    cleanup_memory,
)
from resource_allocation.agent_manager import BOAgent, ResourceEvent
from resource_allocation.qehvi_first_policy import QEHVIFirstDecisionMaker
from resource_allocation.adaptive_scout_policy import AdaptiveScoutDecisionMaker
from resource_allocation.fixed_policy import FixedMixedPolicy, FixedExploitPolicy
from resource_allocation.truth_interface import TruthModelEvaluator

DTYPE       = torch.double
MAX_RETRIES = 3

# Paper-named strategies only — no UCB variants
STRATEGY_CONFIGS = [
    {"type": "qehvi_first",    "name": "Rule-Swap"},
    {"type": "adaptive_scout", "name": "Adaptive-Beta"},
    {"type": "fixed_mixed",    "name": "Fixed-Mixed"},
    {"type": "qEHVI",          "name": "qEHVI"},
]

# Canonical seeds per experiment — same values as the original runs, hardcoded here
# so there is no implicit dependency on results_original/ or any prior results directory.
SEEDS = {
    "Magnets-MAIN":   [282, 1089, 4029, 6485, 7447, 10274, 14745, 14987, 15659, 16295,
                       17003, 30712, 38731, 39199, 40379, 40512, 41444, 47385, 52926, 55595,
                       66301, 66535, 83276, 90511, 93171],
    "Magnets-TIME":   [2489, 9596, 12501, 12814, 25323, 38200, 39630, 40547, 41330, 47225,
                       53387, 59789, 62022, 63261, 69862, 70661, 75745, 77666, 78314, 81098,
                       83744, 85818, 86978, 89924, 98971],
    "Magnets-BUDGET": [341, 6484, 12552, 16706, 22071, 22329, 22980, 23560, 24851, 33709,
                       33924, 41739, 49429, 53758, 54662, 58215, 62407, 65210, 70141, 71862,
                       83179, 86834, 94128, 98219, 99972],
    "Magnets-DUAL":   [11228, 13596, 14882, 19259, 20304, 23904, 30244, 35135, 38277, 38559,
                       40018, 43298, 43796, 45255, 55001, 58217, 58529, 66315, 70674, 71520,
                       74598, 79434, 92591, 93573, 95736],
    "MP-D_MAIN":      [2024, 3741, 6788, 7138, 10918, 19495, 22328, 29842, 31179, 31634,
                       35317, 44221, 47582, 51962, 55083, 57729, 63124, 63801, 74270, 83073,
                       87404, 89217, 92298, 93411, 98635],
    "MP-D_TIME":      [1066, 2183, 7743, 36835, 39324, 42435, 43317, 44984, 56101, 57514,
                       60423, 65958, 65965, 68788, 72050, 73479, 74430, 74859, 78333, 78841,
                       90150, 92508, 96757, 98179, 98380],
    "MP-D_BUDGET":    [3935, 4951, 5124, 7706, 7760, 16281, 16867, 18716, 25109, 29644,
                       31035, 38267, 42605, 46766, 63916, 63993, 64702, 70278, 73119, 74704,
                       76627, 78243, 81034, 83316, 87031],
    "MP-D_DUAL":      [3918, 6754, 12119, 12650, 21497, 25748, 30733, 35415, 36628, 43818,
                       44243, 49262, 56772, 60129, 65239, 74080, 79073, 79834, 79898, 81820,
                       86638, 87347, 89730, 94847, 97032],
}


# ============================================================================
# Experiment configs
# ============================================================================

from resource_allocation.experiments import config

# output_base_dir is overridden at runtime via --output-dir
EXPERIMENTS = {
    name: {**cfg, "create_vis": False, "create_gif": False}
    for name, cfg in config.EXPERIMENTS.items()
}


# ============================================================================
# Helpers
# ============================================================================

def is_cuda_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "cuda" in msg
        or "illegal memory" in msg
        or "device-side assert" in msg
        or "cudaerror" in msg
    )


def is_strategy_complete(seed_dir: str, sname: str, n_iters: int) -> bool:
    """Return True if this seed×strategy already has a complete convergence CSV.

    A complete run logs: 1 header row + 1 init row (iter 0) + n_iters BO rows.
    We require at least n_iters + 1 rows so a run killed after the last iteration
    but before cleanup still counts as done.
    """
    conv_path = os.path.join(seed_dir, sname, f"{sname}_convergence.csv")
    if not os.path.exists(conv_path):
        return False
    with open(conv_path) as f:
        row_count = sum(1 for line in f if line.strip())
    return row_count >= n_iters + 1


# ============================================================================
# Strategy factory
# ============================================================================

def build_strategy(strategy_config: dict, cfg: dict, agent_log_dir: str):
    stype = strategy_config["type"]
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if stype == "qehvi_first":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=cfg["agent_model"],
            temperature=cfg["agent_temperature"],
            api_key=openai_api_key,
        )
        dm = QEHVIFirstDecisionMaker(
            llm=llm,
            problem_description=cfg["problem_description"],
            objective_names=cfg["objective_names"],
            log_dir=agent_log_dir,
        )
    elif stype == "adaptive_scout":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=cfg["agent_model"],
            temperature=cfg["agent_temperature"],
            api_key=openai_api_key,
        )
        dm = AdaptiveScoutDecisionMaker(
            llm=llm,
            problem_description=cfg["problem_description"],
            objective_names=cfg["objective_names"],
            log_dir=agent_log_dir,
            major_elements=cfg.get("major_elements"),
            all_elements=cfg.get("all_elements"),
        )
    elif stype == "fixed_mixed":
        dm = FixedMixedPolicy(beta=2.0)
    elif stype == "qEHVI":
        dm = FixedExploitPolicy(beta=2.0)
    else:
        raise ValueError(f"Unknown strategy type: {stype}")

    agent = BOAgent(
        decision_maker=dm,
        log_dir=agent_log_dir,
        problem_description=cfg["problem_description"],
        objective_names=cfg["objective_names"],
    )
    return agent, dm, stype


# ============================================================================
# Main runner
# ============================================================================

def run_experiment(cfg: dict, output_base: str, dry_run: bool = False, seed_filter=None):
    exp_name   = cfg["name"]
    output_dir = os.path.join(output_base, exp_name)

    if not dry_run and os.getenv("OPENAI_API_KEY") is None:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    if exp_name not in SEEDS:
        raise ValueError(f"No hardcoded seeds for experiment '{exp_name}'. "
                         f"Available: {list(SEEDS.keys())}")

    seeds = list(SEEDS[exp_name])
    if seed_filter:
        seed_set = set(seed_filter)
        seeds = [s for s in seeds if s in seed_set]
        missing = seed_set - set(SEEDS[exp_name])
        if missing:
            print(f"Warning: requested seeds not in canonical list for {exp_name}: {missing}")
        if not seeds:
            raise ValueError(f"None of the requested seeds {seed_filter} found in {exp_name}")

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"Device:     {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Strategies: {[s['name'] for s in STRATEGY_CONFIGS]}")
    print(f"Output:     {output_dir}")
    print(f"{'='*70}")

    if dry_run:
        print("\nDRY RUN — not executing.")
        return

    # ---- Shared setup -------------------------------------------------------
    np.random.seed(cfg["setup_seed"])
    torch.manual_seed(cfg["setup_seed"])

    design_space = DesignSpace(
        components=cfg["components"],
        step=cfg["step"],
        seed=cfg["setup_seed"],
        groups=cfg.get("groups"),
    )
    bounds_t = torch.tensor(design_space.get_bounds(), dtype=DTYPE)
    print(f"\nDesign space: {len(design_space.space)} compositions")

    normalization_params = compute_normalization_params(
        design_space=design_space,
        model_path=cfg["model_path"],
        x_scaler_path=cfg["x_scaler_path"],
        y_scaler_path=cfg["y_scaler_path"],
        postprocess_fn=cfg["postprocess_fn"],
        n_samples=100,
        objective_names=cfg["objective_names"],
    )

    print("[RefPoint] Evaluating full design space for reference point...")
    ev = TruthModelEvaluator(
        model_path=cfg["model_path"],
        x_scaler_path=cfg["x_scaler_path"],
        y_scaler_path=cfg["y_scaler_path"],
        postprocess_outputs=cfg["postprocess_fn"],
        normalize_outputs=False,
    )
    Y_full = ev.evaluate(design_space.space).y
    ev.cleanup()
    del ev
    fixed_ref = np.min(Y_full, axis=0) - cfg["ref_point_margin"]
    print(f"[RefPoint] {fixed_ref}")

    # ---- Per-seed loop ------------------------------------------------------
    failed_log = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'#'*70}")
        print(f"SEED {seed_idx+1}/{len(seeds)}: {seed}")
        print(f"{'#'*70}")

        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        X0, Y0_raw, Y0_norm = generate_shared_initialization(
            design_space=design_space,
            model_path=cfg["model_path"],
            x_scaler_path=cfg["x_scaler_path"],
            y_scaler_path=cfg["y_scaler_path"],
            postprocess_fn=cfg["postprocess_fn"],
            normalization_params=normalization_params,
            n_init=cfg["init_n"],
            seed=seed,
        )
        events = cfg["events_fn"](cfg) if cfg["events_fn"] is not None else None

        for strategy_config in STRATEGY_CONFIGS:
            sname = strategy_config["name"]

            if is_strategy_complete(seed_dir, sname, cfg["iters"]):
                print(f"\n  [skip] {sname} — convergence CSV already complete")
                sys.stdout.flush()
                continue

            print(f"\n  [run]  {sname}...")
            sys.stdout.flush()

            agent_log_dir = os.path.join(seed_dir, sname, "agent_logs")
            os.makedirs(agent_log_dir, exist_ok=True)

            success    = False
            last_error = None

            for attempt in range(MAX_RETRIES + 1):
                agent, dm = None, None
                try:
                    agent, dm, stype = build_strategy(strategy_config, cfg, agent_log_dir)

                    run_bo_experiment(
                        experiment_name=sname,
                        strategy=agent,
                        output_dir=seed_dir,
                        model_path=cfg["model_path"],
                        x_scaler_path=cfg["x_scaler_path"],
                        y_scaler_path=cfg["y_scaler_path"],
                        postprocess_fn=cfg["postprocess_fn"],
                        score_fn=cfg["score_fn"],
                        design_space=design_space,
                        bounds=bounds_t,
                        X0_init=X0,
                        Y0_init_raw=Y0_raw,
                        Y0_init_normalized=Y0_norm,
                        normalization_params=normalization_params,
                        exploration_beta=cfg["beta_explore"],
                        n_iterations=cfg["iters"],
                        mc_samples=cfg["mc_samples"],
                        total_batch_size=cfg["total_batch_size"],
                        pool_subsample=cfg["pool_subsample"],
                        total_budget=cfg["total_budget"],
                        total_time=cfg["total_time"],
                        cost_per_point=cfg["cost_per_point"],
                        time_per_iteration=cfg["time_per_iteration"],
                        objective_names=cfg["objective_names"],
                        objective_display_names=cfg["objective_display"],
                        score_name=cfg["score_name"],
                        seed=seed,
                        use_discrete=cfg["use_discrete"],
                        create_visualization=False,
                        create_gif=False,
                        events=events,
                        pass_uncertainty_to_agent=False,
                        fixed_reference_point=fixed_ref,
                    )

                    success = True
                    break

                except RuntimeError as e:
                    last_error = e
                    agent, dm = None, None  # release refs; finally block checks None
                    cleanup_memory()

                    if is_cuda_error(e) and attempt < MAX_RETRIES:
                        wait = 5 * (attempt + 1)
                        print(
                            f"  [CUDA error] attempt {attempt+1}/{MAX_RETRIES+1} "
                            f"— retrying in {wait}s\n  Error: {e}"
                        )
                        sys.stdout.flush()
                        time.sleep(wait)
                    else:
                        break

                except Exception as e:
                    last_error = e
                    agent, dm = None, None  # release refs; finally block checks None
                    cleanup_memory()
                    break

                finally:
                    if agent is not None:
                        del agent
                    if dm is not None:
                        del dm

            if success:
                cleanup_memory()
                print(f"  Done: {sname}")
            else:
                cleanup_memory()
                err_type = type(last_error).__name__ if last_error else "unknown"
                msg = (
                    f"seed={seed} exp={exp_name} strategy={sname} "
                    f"attempts={min(attempt+1, MAX_RETRIES+1)} "
                    f"error={err_type}: {last_error}"
                )
                print(f"  !! FAILED: {msg}")
                failed_log.append(msg)

        print(f"\nCompleted seed {seed_idx+1}/{len(seeds)}: {seed}")

    # ---- Write failure log --------------------------------------------------
    if failed_log:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "failed_strategies.log")
        with open(log_path, "a") as fh:
            fh.write("\n".join(failed_log) + "\n")
        print(f"\n!! {len(failed_log)} failure(s) logged to {log_path}")
        for m in failed_log:
            print(f"   {m}")
    else:
        print("\nNo failures.")

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE: {exp_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=list(EXPERIMENTS.keys()),
        help="Which experiment to run",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Base directory for results (e.g. results_magnets_v3). "
             "Results land in DIR/<experiment>/seed_<N>/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print seed/strategy plan without running",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        metavar="SEED",
        help="Run only these specific seeds (must be in the hardcoded list for this experiment)",
    )
    cli = parser.parse_args()

    output_base = os.path.abspath(cli.output_dir)

    cfg = dict(EXPERIMENTS[cli.experiment])
    run_experiment(cfg, output_base=output_base, dry_run=cli.dry_run, seed_filter=cli.seeds)
