import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import roc_auc_score

from sevs.utils.metrics import roc_auc


def test_all_scores_tied_gives_chance():
    y = np.array([0, 1, 0, 1])
    assert roc_auc(y, np.full(4, 0.5)) == pytest.approx(0.5)


def test_partial_ties_match_reference():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.5, 0.5, 0.9, 0.1])
    assert roc_auc(y, s) == pytest.approx(0.875)


def test_perfect_and_reversed_ranking():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y, s) == 1.0
    assert roc_auc(1 - y, s) == 0.0


def test_single_class_is_nan():
    assert np.isnan(roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))
    assert np.isnan(roc_auc(np.array([0, 0]), np.array([0.1, 0.2])))


@settings(deadline=None)
@given(
    st.lists(st.tuples(st.integers(0, 1), st.integers(0, 5)), min_size=2, max_size=40)
)
def test_matches_sklearn_with_heavy_ties(pairs):
    y = np.array([p[0] for p in pairs])
    s = np.array([p[1] / 5 for p in pairs])
    if y.min() == y.max():
        assert np.isnan(roc_auc(y, s))
    else:
        assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))
