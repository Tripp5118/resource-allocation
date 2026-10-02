"""
Experiment config registry. The `EXPERIMENTS` dict is the single source of truth for
per-problem settings: ground-truth model paths, design space, BO parameters, resource
budgets, postprocess/score functions, and event schedules, for both alloy systems'
8 experiment variants.

No runnable entry point of its own; `run_campaign.py` imports it.

Model paths below are relative to the repository root, so run from there (replicate.sh
changes directory for you).

Available experiments:
    MP-D_MAIN       Melting Point / Density, no events, 55 iters, 25 seeds
    MP-D_TIME       same + time reduction at iter 15
    MP-D_BUDGET     same + budget cut at iter 15
    MP-D_DUAL       same + time (iter 5) and budget (iter 7) events
    Magnets-MAIN    Permanent magnets, no events, 50 iters, 25 seeds
    Magnets-TIME    same + time reduction at iter 13
    Magnets-BUDGET  same + budget cut at iter 18
    Magnets-DUAL    same + time (iter 13) and budget (iter 18) events
"""

import numpy as np

from resource_allocation.agent_manager import ResourceEvent


# ============================================================================
# POSTPROCESS / SCORE FUNCTIONS
# ============================================================================

def postprocess_mpd(preds_raw: np.ndarray) -> np.ndarray:
    mp = preds_raw[:, 0]
    d  = -1 * np.abs(preds_raw[:, 1])
    return np.column_stack([mp, d])


def score_mpd(Y: np.ndarray) -> np.ndarray:
    return Y[:, 0] / np.abs(Y[:, 1] + 1e-12)


def postprocess_magnets(preds_raw: np.ndarray) -> np.ndarray:
    Ms       = preds_raw[:, 0]
    NegLogHc = -np.log(np.abs(preds_raw[:, 1]) + 1e-6)
    LogHv    = np.log(preds_raw[:, 5] + 1e-6)
    return np.column_stack([Ms, NegLogHc, LogHv])


def score_magnets(Y: np.ndarray) -> np.ndarray:
    return Y[:, 0] * Y[:, 2] / (np.abs(Y[:, 1]) + 1e-12)


# ============================================================================
# EXPERIMENT REGISTRY
# ============================================================================

_MPD_BASE = dict(
    # Paths
    model_path    = "ground_truth_models/TiVNbMoHfTaW-MeltingVsDensity/models/RFR_best_model.pkl",
    x_scaler_path = "ground_truth_models/TiVNbMoHfTaW-MeltingVsDensity/models/x_scaler.pkl",
    y_scaler_path = "ground_truth_models/TiVNbMoHfTaW-MeltingVsDensity/models/y_scaler.pkl",
    # Design space
    components = [
        ("Ti", 0.0,   0.35),
        ("V",  0.0,   0.50),
        ("Nb", 0.225, 0.675),
        ("Mo", 0.0,   0.125),
        ("Hf", 0.0,   0.125),
        ("Ta", 0.0,   0.45),
        ("W",  0.0,   0.125),
    ],
    groups       = None,
    step         = 0.05,
    use_discrete = True,
    # BO params
    init_n           = 5,
    iters            = 55,
    mc_samples       = 256,
    pool_subsample   = 5000,
    total_batch_size = 5,
    # Resources
    total_budget       = 28000.0,
    total_time         = 56.0,
    cost_per_point     = 100.0,
    time_per_iteration = 1.0,
    # Agent
    agent_model       = "gpt-4o",
    agent_temperature = 0.2,
    # Objectives
    objective_names   = ["Melting Point (K)", "Density (g/cm3)"],
    objective_display = ["Melting Point (K)", "Density (g/cm3)"],
    score_name        = "Melting Point / Density (K cm3 g-1)",
    problem_description = (
        "Refractory High Entropy Alloy (RHEA) optimization for high melting point and low density.\n\n"
        "Objectives:\n- Minimize Density\n- Maximize Melting Point\n\n"
        "Goal: Discover a large pareto front of optimal alloys as quickly as possible.\n\n"
        "Input space: 7D compositions, [Ti, V, Nb, Mo, Hf, Ta, W]\n\n"
        "You should balance exploration and optimization given the qEHVI and Mutual Information "
        "acquisition values for each option to best reach the goal within budget and time constraints."
    ),
    # Visualization
    create_vis = True,
    create_gif = True,
    # Run params
    num_seeds            = 25,
    beta_explore         = 2.0,
    setup_seed           = 42,
    output_base_dir      = "./results",
    ref_point_margin     = 0.1,
    # AdaptiveScout element config (all equal contributors for RHEA, no major/minor split)
    major_elements = None,
    all_elements   = ["Ti", "V", "Nb", "Mo", "Hf", "Ta", "W"],
    # Functions
    postprocess_fn = postprocess_mpd,
    score_fn       = score_mpd,
    # Events (override per variant)
    events_fn = None,
)

_MAGNETS_BASE = dict(
    # Paths
    model_path    = "ground_truth_models/FeCoNiVMoCrCuMnCWTaNbAlTiSi-Magnets/models/RFR_best_model.pkl",
    x_scaler_path = "ground_truth_models/FeCoNiVMoCrCuMnCWTaNbAlTiSi-Magnets/models/x_scaler.pkl",
    y_scaler_path = "ground_truth_models/FeCoNiVMoCrCuMnCWTaNbAlTiSi-Magnets/models/y_scaler.pkl",
    # Design space
    components = [
        ("Fe", 0.0, 1.0),   ("Co", 0.0, 1.0),   ("Ni", 0.0, 1.0),
        ("V",  0.0, 0.05),  ("Mo", 0.0, 0.05),  ("Cr", 0.0, 0.0),
        ("Cu", 0.0, 0.0),   ("Mn", 0.0, 0.0),   ("C",  0.0, 0.05),
        ("W",  0.0, 0.0),   ("Ta", 0.0, 0.0),   ("Nb", 0.0, 0.05),
        ("Al", 0.0, 0.0),   ("Ti", 0.0, 0.05),  ("Si", 0.0, 0.05),
    ],
    groups = [
        ("majors",  ["Fe", "Co", "Ni"],                 0.90, 1.00),
        ("dopants", ["V", "Mo", "C", "Nb", "Ti", "Si"], 0.00, 0.10),
    ],
    step         = 0.02,
    use_discrete = True,
    # BO params
    init_n           = 5,
    iters            = 50,
    mc_samples       = 128,
    pool_subsample   = 5000,
    total_batch_size = 5,
    # Resources
    total_budget       = 25500.0,
    total_time         = 51.0,
    cost_per_point     = 100.0,
    time_per_iteration = 1.0,
    # Agent
    agent_model       = "gpt-4o",
    agent_temperature = 0.3,
    # Objectives
    objective_names   = ["Ms", "NegLogHc", "LogHv"],
    objective_display = ["Ms (T)", "-LogHc", "LogHv"],
    score_name        = "Ms x LogHv / LogHc",
    problem_description = (
        "Multi-objective optimization campaign for permanent magnet alloys.\n"
        "Discover a broad Pareto front maximizing all objectives simultaneously.\n\n"
        "Objectives:\n"
        "- Maximize Ms (saturation magnetization)\n"
        "- Maximize NegLogHc (minimize coercivity, i.e. maximize -LogHc)\n"
        "- Maximize LogHv (hardness)\n\n"
        "Your task: balance exploiting known good regions (via qEHVI) with scouting "
        "unexplored design-space regions (via qUCB) to find productive territory "
        "qEHVI has not yet been directed toward."
    ),
    # Visualization
    create_vis = False,
    create_gif = False,
    # Run params
    num_seeds            = 25,
    beta_explore         = 2.0,
    setup_seed           = 42,
    output_base_dir      = "./results",
    ref_point_margin     = 0.1,
    # AdaptiveScout element config — full 15-element list in COMPONENTS order so
    # indices correctly map to pareto_X columns; zeros filtered in _fmt_pareto_coverage
    major_elements = ["Fe", "Co", "Ni"],
    all_elements   = ["Fe", "Co", "Ni", "V", "Mo", "Cr", "Cu", "Mn", "C", "W", "Ta", "Nb", "Al", "Ti", "Si"],
    # Functions
    postprocess_fn = postprocess_magnets,
    score_fn       = score_magnets,
    # Events (override per variant)
    events_fn = None,
)


def _mpd_time_events(cfg):
    return [ResourceEvent(
        iteration=15,
        event_type="time_change",
        description="Time reduced at iteration 15 to 10 time units total",
        modifier=lambda t, v=10: v,
    )]

def _mpd_budget_events(cfg):
    remaining = 10 * cfg["total_batch_size"] * cfg["cost_per_point"]
    return [ResourceEvent(
        iteration=15,
        event_type="budget_change",
        description=f"Budget cut at iteration 15: only 10 iterations remain (${remaining:.0f})",
        modifier=lambda b, v=remaining: v,
    )]

def _mpd_dual_events(cfg):
    budget_remaining = 3 * cfg["total_batch_size"] * cfg["cost_per_point"]
    return [
        ResourceEvent(
            iteration=5,
            event_type="time_change",
            description="Time reduced at iteration 5 to 15 time units total",
            modifier=lambda t, v=15.0: v,
        ),
        ResourceEvent(
            iteration=7,
            event_type="budget_change",
            description=f"Budget cut at iteration 7: only 3 iterations remain (${budget_remaining:.0f})",
            modifier=lambda b, v=budget_remaining: v,
        ),
    ]

def _magnets_time_events(cfg):
    return [ResourceEvent(
        iteration=13,
        event_type="time_change",
        description="Time reduced at iteration 13 to 12 time units total",
        modifier=lambda t, v=12: v,
    )]

def _magnets_budget_events(cfg):
    remaining = 8 * cfg["total_batch_size"] * cfg["cost_per_point"]
    return [ResourceEvent(
        iteration=18,
        event_type="budget_change",
        description=f"Budget cut at iteration 18: only 8 iterations remain (${remaining:.0f})",
        modifier=lambda b, v=remaining: v,
    )]

def _magnets_dual_events(cfg):
    budget_remaining = 8 * cfg["total_batch_size"] * cfg["cost_per_point"]
    return [
        ResourceEvent(
            iteration=13,
            event_type="time_change",
            description="Time reduced at iteration 13 to 36 time units total",
            modifier=lambda t, v=36: v,
        ),
        ResourceEvent(
            iteration=18,
            event_type="budget_change",
            description=f"Budget cut at iteration 18: only 8 iterations remain (${budget_remaining:.0f})",
            modifier=lambda b, v=budget_remaining: v,
        ),
    ]


EXPERIMENTS = {
    "MP-D_MAIN":     {**_MPD_BASE,     "name": "MP-D_MAIN",     "events_fn": None},
    "MP-D_TIME":     {**_MPD_BASE,     "name": "MP-D_TIME",     "events_fn": _mpd_time_events},
    "MP-D_BUDGET":   {**_MPD_BASE,     "name": "MP-D_BUDGET",   "events_fn": _mpd_budget_events},
    "MP-D_DUAL":     {**_MPD_BASE,     "name": "MP-D_DUAL",     "events_fn": _mpd_dual_events},
    "Magnets-MAIN":  {**_MAGNETS_BASE, "name": "Magnets-MAIN",  "events_fn": None},
    "Magnets-TIME":  {**_MAGNETS_BASE, "name": "Magnets-TIME",  "events_fn": _magnets_time_events},
    "Magnets-BUDGET":{**_MAGNETS_BASE, "name": "Magnets-BUDGET","events_fn": _magnets_budget_events},
    "Magnets-DUAL":  {**_MAGNETS_BASE, "name": "Magnets-DUAL",  "events_fn": _magnets_dual_events},
}
