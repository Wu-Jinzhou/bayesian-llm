from __future__ import annotations

import random
from typing import Sequence

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_sequence(*, n_x: int, n_total: int, x: str = "X", y: str = "Y") -> list[str]:
    if n_total < 0 or n_x < 0 or n_x > n_total:
        raise ValueError("Require 0 <= n_x <= n_total.")
    return [x] * n_x + [y] * (n_total - n_x)


def permute_sequence(seq: Sequence[str], rng: np.random.Generator | None = None) -> list[str]:
    if rng is None:
        rng = np.random.default_rng()
    seq = list(seq)
    rng.shuffle(seq)
    return seq

