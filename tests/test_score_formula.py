import numpy as np
import pytest


def _ratio_score(Y: np.ndarray) -> np.ndarray:
    """Reference implementation of the ratio formula used in all Ms-Hc-* runners.
    Y columns: [Ms, NegLogHc, obj3]  (NegLogHc = -LogHc)
    score = Ms * obj3 / LogHc = col0 * col2 / max(-col1, 1e-8)
    """
    return Y[:, 0] * Y[:, 2] / np.maximum(-Y[:, 1], 1e-8)


def test_ratio_formula_correct_value():
    # Ms=1, NegLogHc=-2 (so LogHc=2), obj3=3 → 1*3/2 = 1.5
    Y = np.array([[1.0, -2.0, 3.0]])
    assert np.isclose(_ratio_score(Y)[0], 1.5)


def test_ratio_formula_zero_negloghc_does_not_divide_by_zero():
    # NegLogHc = 0 → epsilon kicks in, result must be finite
    Y = np.array([[1.0, 0.0, 3.0]])
    result = _ratio_score(Y)
    assert np.isfinite(result[0])
    assert result[0] > 0


def test_ratio_formula_positive_negloghc_uses_epsilon():
    # NegLogHc > 0 means LogHc < 0, which is physically unrealistic but
    # the guard must still prevent negative denominators.
    Y = np.array([[1.0, 1.0, 3.0]])
    result = _ratio_score(Y)
    assert np.isfinite(result[0])
    assert result[0] > 0


def test_ratio_formula_batch():
    Y = np.array([
        [1.0, -2.0, 4.0],   # 1*4/2 = 2.0
        [2.0, -4.0, 2.0],   # 2*2/4 = 1.0
    ])
    result = _ratio_score(Y)
    assert np.allclose(result, [2.0, 1.0])
