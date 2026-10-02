from resource_allocation.decision_state import DecisionState, build_strategy_outcome
import pytest


def test_decision_state_accepts_objective_names_list():
    state = DecisionState(
        iteration=1,
        budget_remaining=1000.0,
        time_remaining=10.0,
        cost_per_point=100.0,
        time_per_point=1.0,
        strategy_outcomes=[],
        objective_names=["Ms", "LogHc", "LogHv"],
        best_objectives=[0.5, -1.2, 0.9],
    )
    assert state.objective_names == ["Ms", "LogHc", "LogHv"]
    assert len(state.best_objectives) == 3


def test_decision_state_default_2_objectives():
    state = DecisionState(
        iteration=1,
        budget_remaining=500.0,
        time_remaining=5.0,
        cost_per_point=50.0,
        time_per_point=1.0,
        strategy_outcomes=[],
    )
    assert len(state.objective_names) == 2


def test_strategy_outcome_has_beta_selected():
    outcome = build_strategy_outcome(
        iteration=1,
        selected_option_idx=2,
        n_exploit=3,
        n_explore=2,
        score_before=0.5,
        score_after=0.6,
        pareto_before={"num_points": 4, "hypervolume": 1.2},
        pareto_after={"num_points": 5, "hypervolume": 1.5, "points_improved": 0},
        beta_selected=5.5,
    )
    assert outcome.beta_selected == 5.5


def test_strategy_outcome_beta_defaults_to_2():
    outcome = build_strategy_outcome(
        iteration=1,
        selected_option_idx=2,
        n_exploit=3,
        n_explore=2,
        score_before=0.5,
        score_after=0.6,
        pareto_before={"num_points": 4, "hypervolume": 1.2},
        pareto_after={"num_points": 5, "hypervolume": 1.5, "points_improved": 0},
    )
    assert outcome.beta_selected == 2.0
