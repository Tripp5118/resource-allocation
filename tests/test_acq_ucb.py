import torch
import numpy as np
import pytest
from resource_allocation.acq_ucb import AcquisitionFunctionManager
from resource_allocation.gp_models import GPModelManager

DTYPE = torch.double


def _fit_model(train_data, n_obj):
    X, Y_full, bounds = train_data
    Y = Y_full[:, :n_obj]
    gpm = GPModelManager(bounds)
    return gpm.fit_model(X, Y), bounds


def test_pareto_front_shape_2obj(train_data_2obj):
    X, Y, bounds = train_data_2obj
    afm = AcquisitionFunctionManager(bounds, n_objectives=2, use_discrete=False)
    pareto_front, ref_point = afm.compute_pareto_front(Y.clone())
    assert pareto_front.shape[1] == 2
    assert ref_point.shape[0] == 2


def test_pareto_front_shape_3obj(train_data_3obj):
    X, Y, bounds = train_data_3obj
    afm = AcquisitionFunctionManager(bounds, n_objectives=3, use_discrete=False)
    pareto_front, ref_point = afm.compute_pareto_front(Y)
    assert pareto_front.shape[1] == 3
    assert ref_point.shape[0] == 3


def test_mutual_info_2obj(train_data_2obj):
    model, bounds = _fit_model(train_data_2obj, 2)
    X_np = train_data_2obj[0][:5].numpy()
    afm = AcquisitionFunctionManager(bounds, n_objectives=2, use_discrete=False)
    mi = afm._evaluate_mutual_information(model, X_np, num_optimal_samples=3)
    assert isinstance(mi, float) and not (mi != mi)  # not NaN


def test_mutual_info_3obj(train_data_3obj):
    model, bounds = _fit_model(train_data_3obj, 3)
    X_np = train_data_3obj[0][:5].numpy()
    afm = AcquisitionFunctionManager(bounds, n_objectives=3, use_discrete=False)
    mi = afm._evaluate_mutual_information(model, X_np, num_optimal_samples=3)
    assert isinstance(mi, float) and not (mi != mi)
