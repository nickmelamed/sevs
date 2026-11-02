import numpy as np
import pytest

from sevs.probes.tta import generate_tta_images


def _img():
    rng = np.random.default_rng(0)
    return rng.integers(20, 200, size=(40, 60, 3), dtype=np.uint8)


def test_first_image_is_the_original():
    img = _img()
    out = generate_tta_images(img, ["hflip"])
    assert len(out) == 2
    assert np.array_equal(out[0], img)


def test_brightness_factor_sets_direction():
    img = _img()
    dark = generate_tta_images(img, ["brightness_0.8"])[1]
    bright = generate_tta_images(img, ["brightness_1.2"])[1]
    assert dark.mean() < img.mean() < bright.mean()


def test_scale_changes_the_image_and_keeps_its_shape():
    img = _img()
    for aug in ["scale_0.9", "scale_1.1"]:
        out = generate_tta_images(img, [aug])[1]
        assert out.shape == img.shape
        assert not np.array_equal(out, img)


def test_scale_factors_differ_from_each_other():
    img = _img()
    a = generate_tta_images(img, ["scale_0.9"])[1]
    b = generate_tta_images(img, ["scale_1.1"])[1]
    assert not np.array_equal(a, b)


def test_default_config_augmentations_are_all_distinct():
    img = _img()
    augs = ["hflip", "scale_0.9", "scale_1.1", "brightness_0.8", "brightness_1.2"]
    out = generate_tta_images(img, augs)[1:]
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            assert not np.array_equal(out[i], out[j]), (augs[i], augs[j])


@pytest.mark.parametrize("aug", ["scale_inf", "scale_nan", "brightness_0", "brightness_-1"])
def test_non_finite_or_non_positive_factor_raises(aug):
    with pytest.raises(ValueError):
        generate_tta_images(_img(), [aug])
