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



def dict_to_id(d : dict) -> str:
    """
    Parse a dictionary and generate a unique id.
    The id will have the initials of every key followed by its value.
    Entries are separated by underscore.

    Example:
        d = {first_key: 0, some_key: True} -> fk0_skTrue

    """
    def make_prefix(key : str) -> str:
        return ''.join(w[0] for w in key.split('_'))

    return '_'.join(
        [f'{make_prefix(k)}{v}' for k, v in d.items()])
