import numpy as np
from sevs.probes.perturbations import apply_perturbation

def test_hflip_shape():
    img = np.zeros((10,20,3), dtype=np.uint8)
    out = apply_perturbation(img, "hflip", 1)
    assert out.shape == img.shape
