"""Define handy-dandy general utilities to be reachable across the whole project.

Routine Listings
---------------
set_rng_seed

get_configs

get_prompts
"""

import numpy as np
import random
from typing import Dict
import yaml
import argparse
import os
import wandb


# https://en.wikipedia.org/wiki/Pairing_function
def cantor_pairing(x, y):
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
     Be careful that the environments rng seed should be fixed in `reset()` method.

     """
    np.random.seed(seed)
    random.seed(seed)


def config_parser(path: str) -> Dict:
    """Read general configurations to execute the runs.

    Parameters
    ----------
    path : str
        Path to read the configurations from it.

    Returns
    -------
    dict
       Configurations

    """
    with open(path) as f:
        params = yaml.load(f.read(), yaml.Loader)
    return params


def arg_parser():
    """Receive the inputs needed from the user via command prompt.

    Returns
    -------
    argparse.Namespace
        parser parameters defined by the user.

    """
    parser = argparse.ArgumentParser(description="Enter your inputs")
    parser.add_argument("--config", default="configs/minigrid_ql.yml", type=str, help="Name of the configs file.")
    parser.add_argument("--online_wandb", action="store_true", help="Run wandb in online mode.")
    args = parser.parse_args()
    return args


def init_wandb(online_mode=False):
    if os.path.exists("configs/api_key.wandb"):
        with open("configs/api_key.wandb", 'r') as f:
            os.environ["WANDB_API_KEY"] = f.read()
            if not online_mode:
                os.environ["WANDB_MODE"] = "offline"
    else:
        if not online_mode:
            os.environ["WANDB_MODE"] = "offline"
        key = input("Please enter your wandb api key then press enter:")
        wandb.login(key=key)
