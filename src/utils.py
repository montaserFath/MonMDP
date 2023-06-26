import numpy as np
import random
from typing import Dict
import yaml
import argparse
import os


# https://en.wikipedia.org/wiki/Pairing_function
def cantor_pairing(x, y):
    """Cantor pairing function to uniquely encode two natural numbers into a single natural number.
    Used for seeding.

    Parameters
    ----------
    x : int
    y : int

    Returns
    -------
    int
        Unique integer computed from x and y.

    """
    return int(0.5 * (x + y) * (x + y + 1) + y)


def set_rng_seed(seed):
    """Set random number generator seed across modules that possess random/stochastic computations.

    Parameters
    ----------
    seed : int
        Value of the seed.

    Returns
    -------
    None

    Notes
    -----
    Be careful that the environments seed should be fixed when calling `reset()`.

    """
    np.random.seed(seed)
    random.seed(seed)


def config_parser(path: str) -> Dict:
    """Parse experiment configuration from file.

    Parameters
    ----------
    path : str
        Path to the configuration file.

    Returns
    -------
    dict
       Dictionary with hyperparameters.

    """
    with open(path) as f:
        params = yaml.load(f.read(), yaml.Loader)
    return params


def arg_parser():
    """Parse inputs received via command prompt.

    Returns
    -------
    argparse.Namespace
        Parsed parameters defined by the user.

    """
    parser = argparse.ArgumentParser(description="Enter your inputs")

    parser.add_argument("--config", default="configs/minigrid_ql.yml",
        type=str,
        help="Name of the configs file.")

    parser.add_argument("--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help='WandB mode. If None, WandB will run in whatever mode is currently set.')

    args = parser.parse_args()

    return args
