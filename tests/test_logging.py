import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from resource_allocation.logging_utils import LoggingManager


def _make_manager(tmpdir, name="test"):
    return LoggingManager(log_dir=str(tmpdir), experiment_name=name)


def test_log_evaluations_3obj_columns(tmp_path):
    mgr = _make_manager(tmp_path)
    X = np.random.dirichlet([1]*5, size=4)
    Y = np.random.rand(4, 3)
    score_fn = lambda Y: Y.sum(axis=1)
    mgr.log_evaluations(
        iteration=1,
        X_new=X,
        Y_new=Y,
        score_fn=score_fn,
        objective_names=["Ms", "LogHc", "LogHv"],
    )
    df = pd.read_csv(mgr.evaluations_path)
    assert "Ms" in df.columns
    assert "LogHc" in df.columns
    assert "LogHv" in df.columns


def test_log_iteration_3obj_convergence_columns(tmp_path):
    mgr = _make_manager(tmp_path)
    X = np.random.dirichlet([1]*5, size=6)
    Y = np.random.rand(6, 3)
    score_fn = lambda Y: Y[:, 0]
    mgr.log_iteration(
        iteration=1,
        X_history=X,
        Y_history=Y,
        n_new_points=3,
        strategy=None,
        score_fn=score_fn,
        objective_names=["Ms", "LogHc", "LogHv"],
    )
    df = pd.read_csv(mgr.convergence_path)
    assert "best_Ms" in df.columns
    assert "best_LogHc" in df.columns
    assert "best_LogHv" in df.columns
    assert "mean_Ms" in df.columns
