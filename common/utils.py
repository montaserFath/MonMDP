import numpy as np
import random
import os
from typing import Dict
import yaml
import argparse


def set_rng_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_configs(path: str) -> Dict:
    with open(path) as f:
        params = yaml.load(f.read(), yaml.Loader)
    return params


def get_prompts():
    parser = argparse.ArgumentParser(description="Enter your inputs")
    parser.add_argument("--configs_name", default="configs.yml", type=str, help="Name of the configs file.")
    parser_params = parser.parse_args()
    return parser_params
