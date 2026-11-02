import random

import numpy as np
import torch

from sevs.utils.seed import seed_everything


def draw():
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_same_seed_gives_same_draws_in_all_three_generators():
    seed_everything(7)
    a = draw()
    seed_everything(7)
    b = draw()
    assert a == b


def test_different_seeds_differ():
    seed_everything(1)
    a = draw()
    seed_everything(2)
    assert a != draw()
