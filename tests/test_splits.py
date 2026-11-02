import pytest
from hypothesis import given
from hypothesis import strategies as st

from sevs.utils.splits import split_image_ids

IDS = [f"img{i}" for i in range(20)]


def test_sizes_follow_the_ratios():
    s = split_image_ids(IDS, 0.7, 0.15, 0.15, seed=0)
    assert len(s["train"]) == 14 and len(s["val"]) == 3 and len(s["test"]) == 3


def test_split_is_deterministic_and_seed_dependent():
    a = split_image_ids(IDS, 0.6, 0.2, 0.2, seed=1)
    assert a == split_image_ids(IDS, 0.6, 0.2, 0.2, seed=1)
    assert a != split_image_ids(IDS, 0.6, 0.2, 0.2, seed=2)


def test_order_of_input_does_not_change_the_split():
    a = split_image_ids(IDS, 0.6, 0.2, 0.2, seed=1)
    assert a == split_image_ids(list(reversed(IDS)), 0.6, 0.2, 0.2, seed=1)


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        split_image_ids(IDS, 0.7, 0.2, 0.2, seed=0)
    with pytest.raises(ValueError):
        split_image_ids(IDS, 1.2, -0.1, -0.1, seed=0)


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError):
        split_image_ids(["a", "a", "b"], 0.6, 0.2, 0.2, seed=0)


@given(st.integers(0, 60), st.integers(0, 10), st.integers(0, 10), st.integers(0, 1000))
def test_parts_are_disjoint_and_cover_everything(n, a, b, seed):
    total = a + b + 1
    ids = [str(i) for i in range(n)]
    s = split_image_ids(ids, a / total, b / total, 1 / total, seed=seed)
    parts = s["train"] + s["val"] + s["test"]
    assert sorted(parts) == sorted(ids)
    assert len(set(parts)) == len(parts)
