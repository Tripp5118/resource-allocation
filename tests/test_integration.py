"""
End-to-end BO loop with synthetic truth + mock LLM.
No GPU, no RFR loading, no API calls. Target: < 60 seconds.
"""
import os
import numpy as np
import torch
import pytest
import resource_allocation.gp_models
import resource_allocation.acq_ucb
import resource_allocation.batched_runs

# Force CPU for all integration tests — avoids qEHVI device mismatch on machines with GPU.
_CPU = torch.device("cpu")
resource_allocation.gp_models.DEVICE = _CPU
resource_allocation.acq_ucb.DEVICE = _CPU
resource_allocation.batched_runs.DEVICE = _CPU

DTYPE = torch.double
MC_SAMPLES_FAST = 16
BATCH_SIZE = 3
N_ITERS = 2
D = 5
N_OBJ = 3


def _synthetic_truth(X: np.ndarray) -> np.ndarray:
    Y = np.zeros((X.shape[0], N_OBJ))
    for k in range(N_OBJ):
        center = np.full(D, (k + 1) / (N_OBJ + 1))
        Y[:, k] = -np.sum((X - center) ** 2, axis=1)
    return Y


def _make_design_space():
    from resource_allocation.design_space import DesignSpace
    components = [(f"x{i}", 0.0, 1.0) for i in range(D)]
    return DesignSpace(components, step=0.25, seed=0)


def test_full_bo_loop_3_objectives(mock_llm, tmp_path):
    from resource_allocation.design_space import DesignSpace
    from resource_allocation.gp_models import GPModelManager
    from resource_allocation.acq_ucb import AcquisitionFunctionManager
    from resource_allocation.agent_manager import BOAgent
    from resource_allocation.llm_decision_maker import MultiStageLLMDecisionMaker

    ds = _make_design_space()
    bounds = torch.stack([torch.zeros(D, dtype=DTYPE), torch.ones(D, dtype=DTYPE)])
    gpm = GPModelManager(bounds)

    # Initial data (5 points)
    X0 = ds.sample(5, method="random")
    Y0_raw = _synthetic_truth(X0)
    y_mean = Y0_raw.mean(axis=0)
    y_std = Y0_raw.std(axis=0) + 1e-8
    Y0_norm = (Y0_raw - y_mean) / y_std

    objective_names = ["Obj0", "Obj1", "Obj2"]
    score_fn = lambda Y: Y.sum(axis=1)

    decision_maker = MultiStageLLMDecisionMaker(
        llm=mock_llm,
        problem_description="Synthetic 3-objective test",
        objective_names=objective_names,
        min_seconds_between_calls=0.0,
    )
    agent = BOAgent(
        decision_maker=decision_maker,
        log_dir=str(tmp_path / "agent_logs"),
        problem_description="Synthetic 3-objective test",
        objective_names=objective_names,
    )

    afm = AcquisitionFunctionManager(
        bounds,
        n_objectives=N_OBJ,
        use_discrete=True,
        design_space=ds,
        exploration_beta=2.0,
    )

    X_hist = X0.copy()
    Y_hist = Y0_norm.copy()

    for iteration in range(1, N_ITERS + 1):
        X_t = torch.tensor(X_hist, dtype=DTYPE)
        Y_t = torch.tensor(Y_hist, dtype=DTYPE)

        model = gpm.fit_model(X_t, Y_t)

        pareto_Y, ref_point = afm.compute_pareto_front(Y_t)
        alloc = afm.compute_all_allocations(
            model=model,
            pareto_front=pareto_Y,
            reference_point=ref_point,
            mc_samples=MC_SAMPLES_FAST,
            total_batch_size=BATCH_SIZE,
        )

        selected_idx, reasoning, selected_points = agent.select_resource_allocation(
            iteration=iteration,
            budget_remaining=10000.0,
            time_remaining=20.0,
            cost_per_point=100.0,
            time_per_point=1.0,
            X_history=X_hist,
            Y_history=Y_hist,
            allocation_results=alloc,
            max_batch_size=BATCH_SIZE,
            score_fn=score_fn,
            pareto_front_info={
                "num_points": len(pareto_Y),
                "hypervolume": 1.0,
                "points_added": 1,
                "points_improved": 0,
            },
        )

        new_X = np.vstack([p for p in selected_points if len(p) > 0])
        new_Y_raw = _synthetic_truth(new_X)
        new_Y_norm = (new_Y_raw - y_mean) / y_std

        X_hist = np.vstack([X_hist, new_X])
        Y_hist = np.vstack([Y_hist, new_Y_norm])

    assert X_hist.shape[0] > 5, "Should have evaluated new points beyond the initial 5"
    assert Y_hist.shape[1] == N_OBJ, f"Expected {N_OBJ} objectives, got {Y_hist.shape[1]}"
    assert not np.isnan(Y_hist).any(), "NaN values in objective history"


def test_run_bo_experiment_objective_names_reach_logger(mock_llm, tmp_path):
    """
    Smoke test: run_bo_experiment must complete 1 iteration without NameError.
    Previously crashed because obj1_name/obj2_name were passed to logger
    methods that only accept objective_names.
    """
    import unittest.mock as mock
    from resource_allocation.batched_runs import run_bo_experiment
    from resource_allocation.fixed_policy import PureExploration
    from resource_allocation.design_space import DesignSpace

    _D = 5
    _N_OBJ = 3
    objective_names = ["Ms", "Hc", "HV"]

    components = [(f"x{i}", 0.0, 1.0) for i in range(_D)]
    ds = DesignSpace(components, step=0.5, seed=0)
    bounds = torch.stack([torch.zeros(_D, dtype=DTYPE), torch.ones(_D, dtype=DTYPE)])

    rng = np.random.default_rng(0)
    X0 = rng.dirichlet(np.ones(_D), size=8).astype(np.float64)
    Y0_raw = np.zeros((8, _N_OBJ))
    for k in range(_N_OBJ):
        center = np.full(_D, (k + 1) / (_N_OBJ + 1))
        Y0_raw[:, k] = -np.sum((X0 - center) ** 2, axis=1)
    y_mean = Y0_raw.mean(axis=0)
    y_std = Y0_raw.std(axis=0) + 1e-8
    Y0_norm = (Y0_raw - y_mean) / y_std
    norm_params = {"mean": y_mean, "std": y_std}

    def score_fn(Y):
        return Y.sum(axis=1)

    from resource_allocation.truth_interface import EvaluationResult

    # Patch TruthModelEvaluator so no real model files are needed.
    with mock.patch("resource_allocation.batched_runs.TruthModelEvaluator") as MockEval:
        instance = MockEval.return_value

        def fake_evaluate(X):
            Y_raw = np.zeros((len(X), _N_OBJ))
            for k in range(_N_OBJ):
                center = np.full(_D, (k + 1) / (_N_OBJ + 1))
                Y_raw[:, k] = -np.sum((X - center) ** 2, axis=1)
            Y_norm = (Y_raw - y_mean) / y_std
            return EvaluationResult(y=Y_norm)

        def fake_denormalize(y_norm):
            return y_norm * y_std + y_mean

        instance.evaluate.side_effect = fake_evaluate
        instance.denormalize.side_effect = fake_denormalize

        X_res, Y_res, logger = run_bo_experiment(
            experiment_name="test_exp",
            strategy=PureExploration(),
            output_dir=str(tmp_path),
            model_path="fake.pkl",
            x_scaler_path="fake_x.pkl",
            y_scaler_path="fake_y.pkl",
            postprocess_fn=None,
            score_fn=score_fn,
            design_space=ds,
            bounds=bounds,
            X0_init=X0,
            Y0_init_raw=Y0_raw,
            Y0_init_normalized=Y0_norm,
            normalization_params=norm_params,
            exploration_beta=2.0,
            n_iterations=1,
            mc_samples=16,
            total_batch_size=2,
            pool_subsample=None,
            total_budget=10000.0,
            total_time=100.0,
            cost_per_point=100.0,
            time_per_iteration=1.0,
            objective_names=objective_names,
            score_name="Score",
            seed=0,
            use_discrete=True,
            create_visualization=False,
            create_gif=False,
            events=None,
        )

    assert X_res.shape[0] > 8, "Should have evaluated new points beyond the initial 8"
    assert Y_res.shape[1] == _N_OBJ


def test_llm_decision_maker_select_beta_returns_valid_float():
    from resource_allocation.llm_decision_maker import MultiStageLLMDecisionMaker
    from resource_allocation.decision_state import DecisionState

    responses = iter([
        "EXPLORATION_EFFECTIVENESS: MEDIUM (confidence: 0.6)\nEXPLOITATION_EFFECTIVENESS: HIGH (confidence: 0.7)\nPROBLEM_STRUCTURE: Good landscape\nPARETO_FRONT_SATURATING: NO (confidence: 0.5)",
        "BETA: 7\nREASONING: moderate exploration needed",
    ])

    class MockLLM:
        def invoke(self, prompt):
            class R:
                content = next(responses)
            return R()

    dm = MultiStageLLMDecisionMaker(llm=MockLLM(), log_dir=None, min_seconds_between_calls=0)
    state = DecisionState(
        iteration=1,
        budget_remaining=100.0,
        time_remaining=5.0,
        cost_per_point=1.0,
        time_per_point=0.1,
        strategy_outcomes=[],
        objective_names=["Ms", "NegLogHc", "LogHv"],
        initial_budget=100.0,
        initial_time=5.0,
    )
    beta = dm.select_beta(state)
    assert isinstance(beta, float)
    assert 1.0 <= beta <= 100.0
    assert beta == 7.0


def test_agent_select_beta_returns_float():
    """BOAgent.select_beta should return a float from the decision maker."""
    from resource_allocation.agent_manager import BOAgent
    from resource_allocation.decision_maker import BalancedPolicyDecisionMaker

    agent = BOAgent(decision_maker=BalancedPolicyDecisionMaker(), log_dir=None)
    X = np.zeros((5, 3))
    Y = np.ones((5, 3))
    beta = agent.select_beta(
        iteration=1,
        budget_remaining=100.0,
        time_remaining=5.0,
        cost_per_point=1.0,
        time_per_point=0.1,
        X_history=X,
        Y_history=Y,
        score_fn=lambda Y: Y[:, 0],
        pareto_front_info={"num_points": 3, "hypervolume": 1.2, "points_added": 1, "points_improved": 0},
    )
    assert isinstance(beta, float)
    assert 1.0 <= beta <= 100.0
    assert beta == 2.0  # BalancedPolicyDecisionMaker returns default 2.0


def test_pareto_history_not_doubled_when_select_beta_called_before_select_resource():
    """
    When select_beta is called before select_resource_allocation for the same iteration,
    pareto_history should only have one new entry (not two).
    """
    from resource_allocation.agent_manager import BOAgent
    from resource_allocation.decision_maker import BalancedPolicyDecisionMaker
    from resource_allocation.acq_ucb import AllocationOption, AllocationResults

    agent = BOAgent(decision_maker=BalancedPolicyDecisionMaker(), log_dir=None)
    X = np.zeros((5, 3))
    Y = np.ones((5, 3))
    pareto_info = {"num_points": 3, "hypervolume": 1.2, "points_added": 1, "points_improved": 0}

    initial_len = len(agent.pareto_history)

    # Simulate what batched_runs.py will do: select_beta first, then select_resource_allocation
    agent.select_beta(
        iteration=1,
        budget_remaining=100.0,
        time_remaining=5.0,
        cost_per_point=1.0,
        time_per_point=0.1,
        X_history=X,
        Y_history=Y,
        pareto_front_info=pareto_info,
    )

    dummy_pts = np.zeros((0, 3))
    opts = [AllocationOption(i, 5-i, dummy_pts, dummy_pts, 0.0, 0.0, 5) for i in range(6)]
    alloc = AllocationResults(options=opts)

    agent.select_resource_allocation(
        iteration=1,
        budget_remaining=100.0,
        time_remaining=5.0,
        cost_per_point=1.0,
        time_per_point=0.1,
        X_history=X,
        Y_history=Y,
        allocation_results=alloc,
        max_batch_size=5,
        score_fn=lambda Y: Y[:, 0],
        pareto_front_info=pareto_info,
    )

    # pareto_history should have exactly 1 new entry, not 2
    assert len(agent.pareto_history) == initial_len + 1


def test_llm_decision_maker_make_decision_uses_cached_stages():
    from resource_allocation.llm_decision_maker import MultiStageLLMDecisionMaker
    from resource_allocation.decision_state import DecisionState
    from unittest.mock import MagicMock

    call_count = [0]
    # Responses: Stage 2 (belief update), Stage 2b (beta), Stage 3 (decision)
    staged_responses = [
        "EXPLORATION_EFFECTIVENESS: MEDIUM (confidence: 0.6)\nEXPLOITATION_EFFECTIVENESS: HIGH (confidence: 0.7)\nPROBLEM_STRUCTURE: Good\nPARETO_FRONT_SATURATING: NO (confidence: 0.5)",
        "BETA: 3\nREASONING: conservative",
        "SELECTED_OPTION: 1\nREASONING: exploitation looks good",
    ]
    resp_iter = iter(staged_responses)

    class MockLLM:
        def invoke(self, prompt):
            call_count[0] += 1
            class R:
                content = next(resp_iter)
            return R()

    dm = MultiStageLLMDecisionMaker(llm=MockLLM(), log_dir=None, min_seconds_between_calls=0)

    # Build a minimal AllocationResults with 6 options for make_decision
    from resource_allocation.acq_ucb import AllocationOption, AllocationResults
    import numpy as np
    dummy_pts = np.zeros((0, 3))
    opts = [AllocationOption(i, 5-i, dummy_pts, dummy_pts, 0.0, 0.0, 5) for i in range(6)]
    alloc = AllocationResults(options=opts)

    state = DecisionState(
        iteration=2,
        budget_remaining=80.0,
        time_remaining=4.0,
        cost_per_point=1.0,
        time_per_point=0.1,
        strategy_outcomes=[],  # no reflection
        allocation_options=alloc,
        objective_names=["Ms", "NegLogHc", "LogHv"],
        initial_budget=100.0,
        initial_time=5.0,
    )

    dm.select_beta(state)            # runs Stage 2 + Stage 2b = 2 LLM calls
    calls_after_beta = call_count[0]

    dm.make_decision(state)          # should only run Stage 3 = 1 more call
    assert call_count[0] == calls_after_beta + 1, (
        f"Expected {calls_after_beta + 1} total calls, got {call_count[0]}. "
        "make_decision should skip Stages 1+2 when cached."
    )
