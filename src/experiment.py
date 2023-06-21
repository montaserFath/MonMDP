import gymnasium as gym
import numpy as np
import wandb

from src.actor import Actor
from src.critic import Critic
from src.utils import set_rng_seed, cantor_pairing


class Experiment():
    def __init__(self, env: gym.Env, actor: Actor, critic: Critic,
                        training_episodes, testing_episodes, testing_frequency, rng_seed):
        self._env = env
        self._actor = actor
        self._critic = critic
        self._training_episodes = training_episodes
        self._testing_episodes = testing_episodes
        self._testing_frequency = testing_frequency
        self._rng_seed = rng_seed

    def train(self):
        wandb.init(
            project='QL demo',
            mode='online',
        )

        set_rng_seed(self._rng_seed)
        self._actor.reset()
        self._critic.reset()

        for ep in range(self._training_episodes):
            if ep % self._testing_frequency == 0:
                episode_return = self.test()
                wandb.log({'test/return': episode_return.mean()}, step=ep, commit=False)

            ep_seed = cantor_pairing(self._rng_seed, ep)
            obs, _ = self._env.reset()
            episode_return = 0
            while True:
                action = self._actor(obs)
                next_obs, reward, term, trunc, info = self._env.step(action)
                self._critic.update(obs, action, reward, term, next_obs)
                episode_return += reward
                if term or trunc:
                    break
                obs = next_obs
            wandb.log({'train/return': episode_return}, step=ep, commit=True)
            self._actor.update()

        wandb.finish()
        self._env.close()

    def test(self):
        episode_returns = np.zeros(self._testing_episodes)
        for ep in range(self._testing_episodes):
            ep_seed = cantor_pairing(self._rng_seed, ep)
            obs, _ = self._env.reset(seed=ep_seed)
            while True:
                action = self._actor(obs)
                next_obs, reward, term, trunc, info = self._env.step(action)
                episode_returns[ep] += reward
                if term or trunc:
                    break
                obs = next_obs
        return episode_returns
