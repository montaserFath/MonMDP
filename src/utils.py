import numpy as np
import random
import os
from typing import Dict
import yaml
import argparse


# https://en.wikipedia.org/wiki/Pairing_function
def cantor_pairing(x, y):
  return int(0.5 * (x + y) * (x + y + 1) + y)


def set_rng_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def config_parser(path: str) -> Dict:
    with open(path) as f:
        params = yaml.load(f.read(), yaml.Loader)
    return params


def arg_parser():
    parser = argparse.ArgumentParser(description="Enter your inputs")
    parser.add_argument("--config", default="configs/taxi_ql.yml", type=str, help="Name of the configs file.")
    args = parser.parse_args()
    return args
