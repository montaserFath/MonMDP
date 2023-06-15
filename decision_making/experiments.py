from common import set_rng_seed
import gymnasium as gym
from .action_selection_strategies import BaseStrategy
from .agents import BaseAgent
from abc import ABC, abstractmethod


class Experiment(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass


class TrainExperiment(Experiment):
    def __init__(self, env: gym.Env, agent: BaseAgent, as_strategy: BaseStrategy, **params):
        self._params = params
        self._env = env
        self._agent = agent
        self._as_strategy = as_strategy
        self._n_runs = self._params["n_runs"]
        self._n_episodes = int(float(self._params["n_episodes"]))
        self._rng_seed = self._params["rng_seed"]
        self._n_steps = self._env.spec.max_episode_steps

    def run(self):
        run_logs = []
        for run in range(self._n_runs):
            set_rng_seed(self._rng_seed + run)
            self._agent.reset()
            self._as_strategy.reset()
            episode_logs = []
            for episode in range(self._n_episodes):
                obs, _ = self._env.reset() # Better not set the rng seed here as it biases the agent to only learn the
                # policy corresponding to that seed when testing
                episode_return = 0
                for step in range(self._n_steps):
                    action = self._as_strategy.select_action(obs)
                    next_obs, reward, term, trunc, info = self._env.step(action)
                    self._agent.update_policy(obs, action, reward, trunc or term, next_obs)
                    episode_return += reward
                    if term or trunc:
                        break
                    obs = next_obs
                episode_logs.append(episode_return)
                self._as_strategy.update()
            run_logs.append(episode_logs)
        self._env.close()
        return run_logs


class TestExperiment(Experiment):
    def __init__(self, env: gym.Env, agent: BaseAgent):
        self._env = env
        self._agent = agent

    def run(self):
        obs, _ = self._env.reset(seed=15)
        episode_return = 0
        done = False
        while not done:
            action = self._agent.policy(obs)
            next_obs, reward, term, trunc, info = self._env.step(action)
            episode_return += reward
            if term or trunc:
                done = True
            obs = next_obs
        print("Test episode return:", episode_return)
        self._env.close()
