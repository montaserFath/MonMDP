import gymnasium as gym
import numpy as np
import wandb
from tqdm import tqdm

from src.actor import Actor
from src.critic import Critic
from src.utils import set_rng_seed, cantor_pairing


class Experiment():
    def __init__(self, env: gym.Env, actor: Actor, critic: Critic,
                 training_episodes, testing_episodes, testing_frequency, rng_seed, log_dir: str, save_log: bool = False):
        self._env = env
        self._actor = actor
        self._critic = critic
        self._training_episodes = training_episodes
        self._testing_episodes = testing_episodes
        self._testing_frequency = testing_frequency
        self._rng_seed = rng_seed
        self._log_dir = log_dir
        self._save_train_log = save_log

    def train(self):
        set_rng_seed(self._rng_seed)
        self._actor.reset()
        self._critic.reset()
        joint_reward = {}
        for ep in tqdm(range(self._training_episodes)):
            if ep % self._testing_frequency == 1:
                self._actor.eval()
                episode_return = self.test()
                self._actor.train()
                wandb.log(
                    {
                        'test/environment_reward': episode_return.mean()
                    },
                    step=ep,
                    commit=False
                )

            ep_seed = cantor_pairing(self._rng_seed, ep)
            obs, _ = self._env.reset(seed=ep_seed)
            episode_return = 0.
            episode_loss = 0.
            next_action = None
            ep_joint_reward = []
            while True:
                action = self._actor(obs) if next_action is None else next_action
                next_obs, reward, term, trunc, _ = self._env.step(action)
                if self._critic._on_policy:
                    next_action = self._actor(next_obs)
                episode_loss += self._critic.update(obs, action, reward, term, next_obs, next_action)
                ep_joint_reward.append(reward)
                episode_return += reward
                if term or trunc:
                    break
                obs = next_obs
            joint_reward[ep] = ep_joint_reward
            wandb.log(
                {
                    'train/environment_reward': episode_return,
                    'train/loss_mdp': episode_loss
                },
                step=ep,
                commit=True
            )
            self._actor.update()
        if self._save_train_log:
            np.save(self._log_dir + "/training_joint_reward_{}.npy".format(self._rng_seed), joint_reward)
        # save Q-table as numpy array
        self._critic.save()
        wandb.finish()
        self._env.close()

    def test(self, render: bool = False, save_results: bool = False):
        episode_return = np.zeros(self._testing_episodes)
        for ep in range(self._testing_episodes):
            ep_seed = cantor_pairing(self._rng_seed, ep)
            obs, _ = self._env.reset(seed=ep_seed)
            while True:
                if render:
                    self._env.render()
                action = self._actor(obs)
                next_obs, reward, term, trunc, _ = self._env.step(action)
                episode_return[ep] += reward
                if term or trunc:
                    break
                obs = next_obs
        return episode_return




class MonExperiment(Experiment):
    def train(self):
        set_rng_seed(self._rng_seed)
        self._actor.reset()
        self._critic.reset()
        joint_reward = {}
        eval_joint_reward = np.zeros((self._training_episodes // self._testing_frequency, self._testing_episodes))
        eval_count = 0
        for ep in tqdm(range(self._training_episodes)):
            if ep > 0 and ep % self._testing_frequency == 0:
                self._actor.eval()
                ep_return_true, ep_return_proxy, ep_return_cost, ep_monitor_action, ep_length, _ = self.test()
                eval_joint_reward[eval_count] = ep_return_true + ep_return_cost
                episode_return_true = ep_return_true.mean()
                episode_return_proxy = np.nanmean(ep_return_proxy)
                episode_return_cost = ep_return_cost.mean()
                self._actor.train()
                wandb.log(
                    {
                        'test/environment_reward': episode_return_true,
                        'test/received_reward': episode_return_proxy,
                        'test/monitor_reward': episode_return_cost,
                        "test/monitor_action": np.mean(ep_monitor_action),
                        "test/number_of_timesteps": np.mean(ep_length),
                        "test/joint_reward": episode_return_true + episode_return_cost,
                    },
                    step=ep,
                    commit=False
                )
                eval_count += 1

            ep_seed = cantor_pairing(self._rng_seed, ep)
            obs, _ = self._env.reset(seed=ep_seed)
            episode_return_true = 0.
            episode_return_proxy = 0.
            episode_return_cost = 0.
            episode_loss_mdp = 0.
            episode_loss_mon = 0.
            reward_seen = False
            episode_monitor_action_count, time_steps = 0, 0
            next_action = None
            ep_joint_reward = []
            while True:
                time_steps += 1
                action = self._actor(obs) if next_action is None else next_action

                if action["monitor"] == 1:
                    episode_monitor_action_count += 1
                next_obs, reward, term, trunc, info = self._env.step(action)
                if self._critic._on_policy:
                    next_action = self._actor(next_obs)
                step_loss_mdp, step_loss_mon = \
                    self._critic.update(obs, action, reward, term, next_obs, next_action)

                episode_return_true += info['mdp_reward']
                episode_return_cost += reward['monitor']
                if not np.isnan(reward['mdp']):
                    reward_seen = True
                    episode_return_proxy += reward['mdp']
                if not np.isnan(step_loss_mdp):
                    episode_loss_mdp += step_loss_mdp
                if not np.isnan(step_loss_mon):
                    episode_loss_mon += step_loss_mon

                ep_joint_reward.append(info['mdp_reward'] + reward['monitor'])

                if term or trunc:
                    if not reward_seen:
                        episode_return_proxy = np.nan
                    break

                obs = next_obs
            joint_reward.update({ep: ep_joint_reward})
            wandb.log(
                {
                    'train/environment_reward': episode_return_true,
                    'train/received_reward': episode_return_proxy,
                    'train/monitor_reward': episode_return_cost,
                    'train/loss_mdp': episode_loss_mdp,
                    'train/loss_mon': episode_loss_mon,
                    "train/monitor_action": episode_monitor_action_count,
                    "train/number_of_timesteps": time_steps,
                    "train/joint_reward": episode_return_true + episode_return_cost,
                },
                step=ep,
                commit=True
            )
            self._actor.update()
        # save Q-table as numpy array
        self._critic.save()
        if self._save_train_log:
            np.save(self._log_dir + "/training_joint_reward_{}.npy".format(self._rng_seed), joint_reward)
            np.save(self._log_dir + "/evaluation_joint_reward_{}.npy".format(self._rng_seed), eval_joint_reward)
        wandb.finish()
        self._env.close()

    def test(self, render: bool = False, save_results: bool = False):
        episode_return_true = np.zeros(self._testing_episodes)
        episode_return_proxy = np.zeros(self._testing_episodes)
        episode_return_cost = np.zeros(self._testing_episodes)
        episode_monitor_action = np.zeros(self._testing_episodes)
        episode_length = np.zeros(self._testing_episodes)
        trajectories = {}
        for ep in range(self._testing_episodes):
            reward_seen = False
            ep_seed = cantor_pairing(self._rng_seed, ep)
            obs, _ = self._env.reset(seed=ep_seed)
            episode_monitor_action_count, time_steps = 0, 0
            ep_states, ep_actions = [], []
            while True:
                ep_states.append([obs["mdp"].item(), obs["monitor"]])
                time_steps += 1
                if render:
                    self._env.render()
                action = self._actor(obs)
                ep_actions.append([action["mdp"], action["monitor"]])
                if action["monitor"] == 1:
                    episode_monitor_action_count += 1
                next_obs, reward, term, trunc, info = self._env.step(action)
                episode_return_true[ep] += info['mdp_reward']
                episode_return_cost[ep] += reward['monitor']
                if not np.isnan(reward['mdp']):
                    reward_seen = True
                    episode_return_proxy[ep] += reward['mdp']
                if term or trunc:
                    if not reward_seen:
                        episode_return_proxy[ep] = np.nan
                    break
                obs = next_obs
            episode_monitor_action[ep] = episode_monitor_action_count
            episode_length[ep] = time_steps
            trajectories[ep] = {
                "states": np.array(ep_states),
                "actions": np.array(ep_actions),
                "environment_reward": episode_return_true,
                "received_reward": episode_return_proxy,
                "monitor_reward": episode_return_cost,
                "joint_reward": episode_return_true + episode_return_cost,
            }
        if save_results:
            np.save(self._log_dir + "/trajectories.npy", trajectories)
        return episode_return_true, episode_return_proxy, episode_return_cost, episode_monitor_action, episode_length, trajectories
