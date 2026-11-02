import numpy as np
import pytest

from sevs.probes.tta import generate_tta_images, generate_tta_views


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


H, W = 40, 60
BOX = np.array([[20.0, 12.0, 40.0, 28.0]])


def view(aug):
    return generate_tta_views(_img(), [aug])[1]


def test_views_start_with_the_untouched_original():
    first = generate_tta_views(_img(), ["hflip"])[0]
    assert np.array_equal(first.image, _img())
    assert first.invert_boxes(BOX).tolist() == BOX.tolist()


def test_hflip_view_maps_boxes_back():
    flipped = BOX.copy()
    flipped[:, 0], flipped[:, 2] = W - BOX[:, 2], W - BOX[:, 0]
    assert view("hflip").invert_boxes(flipped) == pytest.approx(BOX)


def test_zoom_in_view_maps_boxes_back():
    f = 0.8
    cw, ch = round(W * f), round(H * f)
    x0, y0 = (W - cw) // 2, (H - ch) // 2
    forward = BOX.copy()
    forward[:, [0, 2]] = (BOX[:, [0, 2]] - x0) * (W / cw)
    forward[:, [1, 3]] = (BOX[:, [1, 3]] - y0) * (H / ch)
    assert view("scale_0.8").invert_boxes(forward) == pytest.approx(BOX)


def test_shrink_view_maps_boxes_back():
    f = 1.25
    sw, sh = round(W / f), round(H / f)
    x0, y0 = (W - sw) // 2, (H - sh) // 2
    forward = BOX.copy()
    forward[:, [0, 2]] = BOX[:, [0, 2]] * (sw / W) + x0
    forward[:, [1, 3]] = BOX[:, [1, 3]] * (sh / H) + y0
    assert view("scale_1.25").invert_boxes(forward) == pytest.approx(BOX)


def test_brightness_view_leaves_boxes_alone():
    assert view("brightness_1.2").invert_boxes(BOX).tolist() == BOX.tolist()


def test_views_images_match_generate_tta_images():
    augs = ["hflip", "scale_0.9", "brightness_1.2"]
    views = generate_tta_views(_img(), augs)
    images = generate_tta_images(_img(), augs)
    assert all(np.array_equal(v.image, i) for v, i in zip(views, images))
