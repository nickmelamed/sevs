import numpy as np

from sevs.models.meta_error import fit_and_score_by_image

CFG = {
    "model": {"family": "logistic_regression", "params": {"C": 1.0, "max_iter": 200}},
    "training": {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2},
}


def make_rows(n_images=30, per_image=4, seed=0):
    rng = np.random.default_rng(seed)
    ids, X, y = [], [], []
    for i in range(n_images):
        for _ in range(per_image):
            label = int(rng.integers(0, 2))
            ids.append(f"img{i}")
            X.append([label + rng.normal(0, 0.5)])
            y.append(label)
    return ids, np.array(X), np.array(y)


def test_no_image_is_in_two_parts():
    ids, X, y = make_rows()
    out = fit_and_score_by_image(X, y, ids, CFG, ["f"], seed=0)
    parts = [set(out.images[k]) for k in ("train", "val", "test")]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
    assert sum(len(p) for p in parts) == 30


def test_test_rows_are_exactly_the_test_images_rows():
    ids, X, y = make_rows()
    out = fit_and_score_by_image(X, y, ids, CFG, ["f"], seed=0)
    expected = sum(i in set(out.images["test"]) for i in ids)
    assert len(out.y_test) == len(out.p_test) == expected


def test_informative_feature_gives_high_test_auroc():
    ids, X, y = make_rows(n_images=60)
    out = fit_and_score_by_image(X, y, ids, CFG, ["f"], seed=0)
    assert out.test.auroc > 0.8


def test_single_class_training_data_gives_nan():
    ids, X, _ = make_rows()
    out = fit_and_score_by_image(X, np.zeros(len(ids), dtype=int), ids, CFG, ["f"], seed=0)
    assert np.isnan(out.test.auroc) and np.isnan(out.val.auroc)
