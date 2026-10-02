import numpy as np
import pytest
import os
from pathlib import Path


def _make_vm(tmp_path, obj_names):
    """Create a minimal VisualizationManager without a design_space (for Pareto-only tests)."""
    # We'll monkey-patch to avoid needing a real design_space
    from unittest.mock import MagicMock
    from resource_allocation.visualization import VisualizationManager

    # Create a mock design_space
    ds = MagicMock()
    ds.space = np.random.dirichlet([1]*5, size=30)  # 30 points on 5D simplex
    ds.labels = [f"x{i}" for i in range(5)]

    vm = VisualizationManager(
        log_dir=str(tmp_path),
        design_space=ds,
        experiment_name="test",
        objective_names=obj_names,
    )
    return vm


def test_pareto_plot_2obj_creates_file(tmp_path):
    vm = _make_vm(tmp_path, ["Ms", "LogHc"])
    Y_all = np.random.rand(20, 2)
    Y_pareto = Y_all[:3]
    vm.create_pareto_front_plot(
        Y_pareto=Y_pareto,
        Y_all=Y_all,
        objective_names=["Ms", "LogHc"],
        save_path=str(tmp_path / "pareto.png"),
        iteration=1,
    )
    assert (tmp_path / "pareto.png").exists()


def test_pareto_plot_3obj_creates_file(tmp_path):
    vm = _make_vm(tmp_path, ["Ms", "LogHc", "LogHv"])
    Y_all = np.random.rand(20, 3)
    Y_pareto = Y_all[:3]
    vm.create_pareto_front_plot(
        Y_pareto=Y_pareto,
        Y_all=Y_all,
        objective_names=["Ms", "LogHc", "LogHv"],
        save_path=str(tmp_path / "pareto3d.png"),
        iteration=1,
    )
    assert (tmp_path / "pareto3d.png").exists()


def test_pareto_plot_4obj_skips_gracefully(tmp_path, capsys):
    vm = _make_vm(tmp_path, ["A", "B", "C", "D"])
    vm.create_pareto_front_plot(
        Y_pareto=np.random.rand(3, 4),
        Y_all=np.random.rand(20, 4),
        objective_names=["A", "B", "C", "D"],
        save_path=str(tmp_path / "skipped.png"),
        iteration=1,
    )
    assert not (tmp_path / "skipped.png").exists()
    captured = capsys.readouterr()
    assert "skipped" in captured.out.lower() or "N>3" in captured.out


# ---------------------------------------------------------------------------
# Tests for _predict_objectives_from_gp and create_objective_space_plot
# These require a real GP fit (CPU, small design space) but no GPU.
# ---------------------------------------------------------------------------

def _make_vm_with_ds(tmp_path, obj_names):
    """VisualizationManager backed by a real tiny DesignSpace for GP-prediction tests."""
    import torch
    from resource_allocation.design_space import DesignSpace
    from resource_allocation.visualization import VisualizationManager

    components = [(f"x{i}", 0.0, 1.0) for i in range(5)]
    ds = DesignSpace(components, step=0.5, seed=0)

    norm_params = {
        "mean": np.zeros(len(obj_names)),
        "std": np.ones(len(obj_names)),
    }

    return VisualizationManager(
        design_space=ds,
        log_dir=str(tmp_path),
        experiment_name="test",
        normalization_params=norm_params,
        objective_names=obj_names,
        major_indices=[0, 1, 2],
    )


def test_predict_objectives_from_gp_returns_correct_shape(tmp_path, train_data_3obj):
    """_predict_objectives_from_gp must return (n_ds, n_obj) in original scale."""
    import torch
    from resource_allocation.design_space import DesignSpace
    from resource_allocation.visualization import VisualizationManager
    from resource_allocation.gp_models import GPModelManager

    X_t, Y_t, bounds = train_data_3obj
    gpm = GPModelManager(bounds)
    model = gpm.fit_model(X_t, Y_t)

    components = [(f"x{i}", 0.0, 1.0) for i in range(5)]
    ds = DesignSpace(components, step=0.5, seed=0)
    n_ds = len(ds.space)

    norm_params = {
        "mean": Y_t.numpy().mean(axis=0),
        "std": Y_t.numpy().std(axis=0) + 1e-8,
    }

    vm = VisualizationManager(
        design_space=ds,
        log_dir=str(tmp_path),
        experiment_name="test",
        normalization_params=norm_params,
        objective_names=["A", "B", "C"],
        major_indices=[0, 1, 2],
    )

    Y_pred = vm._predict_objectives_from_gp(model, bounds)
    assert Y_pred.shape == (n_ds, 3)
    assert np.isfinite(Y_pred).all()


def test_create_objective_space_plot_with_model_creates_file(tmp_path, train_data_3obj):
    """create_objective_space_plot must accept model+bounds and produce a PNG."""
    import torch
    from resource_allocation.design_space import DesignSpace
    from resource_allocation.visualization import VisualizationManager
    from resource_allocation.gp_models import GPModelManager

    X_t, Y_t, bounds = train_data_3obj
    gpm = GPModelManager(bounds)
    model = gpm.fit_model(X_t, Y_t)

    components = [(f"x{i}", 0.0, 1.0) for i in range(5)]
    ds = DesignSpace(components, step=0.5, seed=0)

    norm_params = {
        "mean": Y_t.numpy().mean(axis=0),
        "std": Y_t.numpy().std(axis=0) + 1e-8,
    }

    vm = VisualizationManager(
        design_space=ds,
        log_dir=str(tmp_path),
        experiment_name="test",
        normalization_params=norm_params,
        objective_names=["A", "B", "C"],
        major_indices=[0, 1, 2],
    )

    X_hist = X_t.numpy()
    Y_hist = Y_t.numpy()
    X_new = X_hist[-2:]

    path = vm.create_objective_space_plot(
        iteration=1,
        X_history=X_hist,
        Y_history=Y_hist,
        X_new=X_new,
        model=model,
        bounds=bounds,
    )
    assert path is not None
    assert Path(path).exists()
