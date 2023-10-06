"""Utility functions"""
import random
import numpy as np


# https://en.wikipedia.org/wiki/Pairing_function
def cantor_pairing(first_int: int, second_int: int) -> int:
    """
    Cantor pairing function to uniquely encode two
    natural numbers into a single natural number.
    Used for seeding.

    Args:
        first_int (int)
        second_int (int)

    Returns:
        A unique integer computed from x and y.

    """
    return int(0.5 * (first_int + second_int) * (first_int + second_int + 1) + second_int)


def set_rng_seed(seed: int = None) -> None:
    """
    Set random number generator seed across modules
    that possess random/stochastic computations.

    Args:
        seed (int)

    """
    np.random.seed(seed)
    random.seed(seed)


def dict_to_id(dictionary: dict) -> str:
    """
    Parse a dictionary and generate a unique id.
    The id will have the initials of every key followed by its value.
    Entries are separated by underscore.

    Example:
        dictionary = {first_key: 0, some_key: True} -> fk0_skTrue

    """

    def make_prefix(key: str) -> str:
        """joint strings together"""
        return "".join(w[0] for w in key.split("_"))

    return "_".join([f"{make_prefix(k)}{v}" for k, v in dictionary.items()])
