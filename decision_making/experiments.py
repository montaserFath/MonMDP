import numpy as np

from common import set_rng_seed
import gymnasium as gym
from .strategies import BaseStrategy
from .agents import BaseAgent


class Experiment:
    def __init__(self, env: gym.Env, agent: BaseAgent, strategy: BaseStrategy, **params):
        self._params = params
        self._env = env
        self._agent = agent
        self._strategy = strategy
        self._n_runs = self._params["n_runs"]
        self._n_episodes = int(float(self._params["n_episodes"]))
        self._rng_seed = self._params["rng_seed"]
        self._n_steps = self._env.spec.max_episode_steps

    def run(self):
        run_logs = []
        for run in range(self._n_runs):
            set_rng_seed(self._rng_seed + run)
            episode_logs = []
            for episode in range(self._n_episodes):
                obs, _ = self._env.reset(seed=self._rng_seed + run)
                episode_return = 0
                for step in range(self._n_steps):
                    action = self._strategy.select_action(obs)
                    next_obs, reward, term, trunc, info = self._env.step(action)
                    self._agent.update_policy(obs, action, reward, trunc or term, next_obs)
                    episode_return += reward
                    if term or trunc:
                        break
                    obs = next_obs
                episode_logs.append(episode_return)
                self._strategy.update()
            run_logs.append(episode_logs)
        return run_logs
