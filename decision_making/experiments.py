from common import RNGSeeder
import gymnasium as gym
from .strategies import BaseStrategy


class Experiment:
    def __init__(self, env: gym.Env, agent, strategy: BaseStrategy, **configs):
        self._configs = configs
        self._env = env
        self.rng_seeder = RNGSeeder(self._env)
        self._agent = agent
        self._strategy = strategy
        self._n_runs = self._configs["n_runs"]
        self._n_episodes = self._configs["n_episodes"]
        self._rng_seed = self._configs["rng_seed"]
        self._n_steps = self._env.spec.max_episode_steps

    def run(self):
        for run in self._n_runs:
            self.rng_seeder.set_seed(self._rng_seed + run)
            for episode in range(self._n_episodes):
                obs = self._env.reset()
                for step in range(self._n_steps):
                    action = self._strategy.select_action(obs)
                    obs, reward, done, info = self._env.step(action)

