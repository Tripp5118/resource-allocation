# tests/conftest.py
"""Shared lightweight fixtures — no GPU, no real models, no LLM API."""
import numpy as np
import torch
import pytest

DTYPE = torch.double
D = 5          # input dimension (5-component simplex)
N_OBJ_2 = 2
N_OBJ_3 = 3
N_TRAIN = 15   # enough for GP to fit without crashing


def _make_simplex_points(n: int, d: int = D, seed: int = 0) -> np.ndarray:
    """Sample n random points from the d-component composition simplex."""
    rng = np.random.default_rng(seed)
    X = rng.dirichlet(np.ones(d), size=n)
    return X.astype(np.float64)


def _synthetic_objectives(X: np.ndarray, n_obj: int) -> np.ndarray:
    """Simple quadratic objectives — no model loading required."""
    Y = np.zeros((X.shape[0], n_obj))
    for k in range(n_obj):
        center = np.full(X.shape[1], (k + 1) / (n_obj + 1))
        Y[:, k] = -np.sum((X - center) ** 2, axis=1)  # maximise
    return Y


@pytest.fixture
def bounds_5d():
    return torch.stack([torch.zeros(D, dtype=DTYPE), torch.ones(D, dtype=DTYPE)])


@pytest.fixture
def train_data_2obj(bounds_5d):
    X = _make_simplex_points(N_TRAIN)
    Y = _synthetic_objectives(X, N_OBJ_2)
    return (
        torch.tensor(X, dtype=DTYPE),
        torch.tensor(Y, dtype=DTYPE),
        bounds_5d,
    )


@pytest.fixture
def train_data_3obj(bounds_5d):
    X = _make_simplex_points(N_TRAIN, seed=1)
    Y = _synthetic_objectives(X, N_OBJ_3)
    return (
        torch.tensor(X, dtype=DTYPE),
        torch.tensor(Y, dtype=DTYPE),
        bounds_5d,
    )


@pytest.fixture
def mock_llm():
    """LLM that returns a hardcoded valid Stage-3 response without any API call."""
    class MockLLM:
        def invoke(self, prompt: str):
            class Resp:
                content = (
                    "EXPLORATION_EFFECTIVENESS: MEDIUM (confidence: 0.6)\n"
                    "EXPLOITATION_EFFECTIVENESS: HIGH (confidence: 0.7)\n"
                    "PROBLEM_STRUCTURE: Landscape appears smooth near current best.\n"
                    "PARETO_FRONT_SATURATING: NO (confidence: 0.5)\n"
                    "SELECTED_OPTION: 0\n"
                    "REASONING: Exploitation is working; focus on it."
                )
            return Resp()
    return MockLLM()
