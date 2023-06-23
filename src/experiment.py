import gymnasium as gym
import numpy as np
import wandb
from tqdm import tqdm

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

        for ep in tqdm(range(1, 1 + self._training_episodes)):
            if ep % self._testing_frequency == 0:
                self._actor.eval()
                episode_return = self.test()
                self._actor.train()
                wandb.log({'test/return': episode_return.mean()}, step=ep, commit=False)

            ep_seed = cantor_pairing(self._rng_seed, ep)
            obs, _ = self._env.reset()
            episode_return = 0
            while True:
                action = self._actor(obs)
                next_obs, reward, term, trunc, info = self._env.step(action)
                if obs[1] == 1:
                    self._critic.update(obs[0], action, reward[0] + reward[1], term[0], next_obs[0]) # what's the use of term[1]
                episode_return += reward[0] if reward[0] is not np.NAN else 0
                if term[0] or trunc[0]:
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
            obs, _ = self._env.reset()
            while True:
                action = self._actor(obs)
                next_obs, reward, term, trunc, info = self._env.step(action)
                episode_returns[ep] += reward[0] if reward[0] is not np.NAN else 0
                if term[0] or trunc[0]:
                    break
                obs = next_obs
        return episode_returns
