import numpy as np
import random
import os


class RNGSeeder:
    def __init__(self, env):
        self._env = env

    def set_seed(self, seed):
        self._env.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
