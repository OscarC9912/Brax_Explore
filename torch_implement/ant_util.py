"""Helper Functions for torch_ant implementation"""

import numpy as np


def random_split(rng: np.ndarray, num: int = 2) -> np.ndarray:
    """Split a PRNG key into num new keys by adding a leading axis"""
    if rng is None:
        raise "Error: rng value cannot be None"
    rng = np.random.default_rng(rng)
    return rng.integers(low=0, high=2**32, dtype='unit32', size=(num, 2))