import torch, pytest
from resource_allocation.gp_models import GPModelManager

DTYPE = torch.double


def test_fit_and_predict_2_objectives(train_data_2obj):
    X, Y, bounds = train_data_2obj
    gpm = GPModelManager(bounds)
    model = gpm.fit_model(X, Y)
    mean, std = gpm.predict(model, X[:4])
    assert mean.shape == (4, 2), f"expected (4,2), got {mean.shape}"
    assert std.shape == (4, 2)
    assert not torch.tensor(mean).isnan().any()


def test_fit_and_predict_3_objectives(train_data_3obj):
    X, Y, bounds = train_data_3obj
    gpm = GPModelManager(bounds)
    model = gpm.fit_model(X, Y)
    mean, std = gpm.predict(model, X[:4])
    assert mean.shape == (4, 3), f"expected (4,3), got {mean.shape}"
    assert std.shape == (4, 3)


def test_model_num_outputs_matches_n_obj(train_data_3obj):
    X, Y, bounds = train_data_3obj
    gpm = GPModelManager(bounds)
    model = gpm.fit_model(X, Y)
    assert model.num_outputs == 3
