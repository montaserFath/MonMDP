import numpy as np
import random


# https://en.wikipedia.org/wiki/Pairing_function
def cantor_pairing(x: int, y: int) -> int:
    """
    Cantor pairing function to uniquely encode two
    natural numbers into a single natural number.
    Used for seeding.

    Args:
        x (int)
        y (int)

    Returns:
        A unique integer computed from x and y.

    """
    return int(0.5 * (x + y) * (x + y + 1) + y)


def set_rng_seed(seed : int = None) -> None:
    """
    Set random number generator seed across modules
    that possess random/stochastic computations.

    Args:
        seed (int)

    """
    np.random.seed(seed)
    random.seed(seed)
