# pylint: disable=too-many-locals, too-many-statements, too-many-instance-attributes, too-many-arguments
"""Experiments wrapper for Training and evaluating algorithms in MDP and Monitor MDP """
import os
import gymnasium as gym
import numpy as np
import wandb
from src.actor import Actor
from src.critic import Critic
from src.utils import set_rng_seed, cantor_pairing
from src.replay_buffer import TorchReplayMemory
from src.policy_analysis import get_action_ind


class Experiment:
    """Run experiments for training and testing in MDP env"""

    def __init__(
        self,
        env: gym.Env,
        actor: Actor,
        critic: Critic,
        training_timesteps,
        testing_episodes,
        testing_frequency,
        rng_seed,
        log_dir: str,
        replay_buffer_size: int,
        save_log: bool = False,
        replay_buffer: bool = False,
        start_train_timestep: int = int(1e4),
    ):
        self._env = env
        self._actor = actor
        self._critic = critic
        self._training_timesteps = training_timesteps
        self._testing_episodes = testing_episodes
        self._testing_frequency = testing_frequency
        self._rng_seed = rng_seed
        self._log_dir = log_dir
        self._save_train_log = save_log
        self._visit_table = None
        self._checkpoint_count = 0
        self._start_train_timestep = start_train_timestep  # start training after reaching number of timesteps
        self.buffer = TorchReplayMemory(max_size=int(replay_buffer_size)) if replay_buffer else None

    def train(self):
        """Train an algorithm in MDP env, logs and save results"""
        set_rng_seed(self._rng_seed)
        self._actor.reset()
        self._critic.reset()
        joint_reward = {}
        total_timesteps = 0
        episode = 0
        while total_timesteps < self._training_timesteps:
            if total_timesteps > self._testing_frequency * self._checkpoint_count:
                self._actor.eval()
                episode_return = self.test()
                self._actor.train()
                wandb.log({"test/environment_reward": episode_return.mean()}, step=episode, commit=False)

            ep_seed = cantor_pairing(self._rng_seed, episode)
            obs, _ = self._env.reset(seed=ep_seed)
            episode_return = 0.0
            episode_loss = 0.0
            next_action = None
            ep_joint_reward = []
            episode_timesteps = 0
            while True:
                episode_timesteps += 1
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
                self._actor.update()
            joint_reward[episode_timesteps] = ep_joint_reward
            total_timesteps += episode_timesteps
            episode += 1
            wandb.log(
                {"train/environment_reward": episode_return, "train/loss_mdp": episode_loss}, step=episode, commit=True
            )
        if self._save_train_log:
            np.save(self._log_dir + "/training_joint_reward_{}.npy".format(self._rng_seed), joint_reward)
        # save Q-table as numpy array
        self._critic.save(seed=self._rng_seed)
        wandb.finish()
        self._env.close()

    def test(self, render: bool = False) -> dict:
        """Evaluate an algorithm in MDP env, logs and save results"""
        test_return = {}
        total_timesteps = 0
        episode = 0
        while episode < self._testing_episodes:
            episode_return = []
            ep_seed = cantor_pairing(self._rng_seed, episode)
            obs, _ = self._env.reset(seed=ep_seed)
            episode_timesteps = 0
            while True:
                episode_timesteps += 1
                if render:
                    self._env.render()
                action = self._actor(obs)
                next_obs, reward, term, trunc, _ = self._env.step(action)
                episode_return.append(reward)
                if term or trunc:
                    break
                obs = next_obs
            test_return.update({episode: np.array(episode_return)})
            episode += 1
            total_timesteps += episode_timesteps
        return test_return


class MonExperiment(Experiment):
    """Run experiments for training and testing in Monitor MDP env"""

    def reset_visit_table(self):
        if isinstance(self._env.observation_space["mdp"], gym.spaces.Discrete):
            mdp_obs_n = self._env.observation_space["mdp"].n
        else:
            mdp_obs_n = 81  # fix this
        mdp_action_n = self._env.action_space["mdp"].n
        mon_obs_n, mon_action_n = self._env.observation_space["monitor"].n, self._env.action_space["monitor"].n
        self._visit_table = np.zeros((mdp_obs_n * mon_obs_n, mdp_action_n * mon_action_n))
        isinstance(self._env.observation_space["mdp"], gym.spaces.Discrete)

    def train(self, checkpoint: bool = True):
        """Train an algorithm in Monitor MDP env, logs and save results"""
        set_rng_seed(self._rng_seed)
        self._actor.reset()
        self._critic.reset()
        joint_reward = {}
        eval_joint_reward = {}
        eval_count = 0
        total_timesteps = 0
        episode = 0
        batch_size = 128
        n_epoches_per_timesteps = 50
        # reset visit table
        self.reset_visit_table()
        while total_timesteps < self._training_timesteps:
            if total_timesteps > self._testing_frequency * self._checkpoint_count:
                # perform/save checkpoint
                self.checkpoint()

                self._actor.eval()
                (
                    ep_return_true,
                    ep_return_proxy,
                    ep_return_cost,
                    ep_monitor_action,
                    ep_length,
                    ep_discount_reward,
                    _,
                ) = self.test()
                eval_joint_reward.update({episode: ep_discount_reward})
                episode_return_true = ep_return_true.mean()
                episode_return_proxy = np.nanmean(ep_return_proxy)
                episode_return_cost = ep_return_cost.mean()
                self._actor.train()
                logs = {
                    "environment_reward": episode_return_true,
                    "received_reward": episode_return_proxy,
                    "monitor_reward": episode_return_cost,
                    "monitor_action": np.mean(ep_monitor_action),
                    "number_of_timesteps": np.mean(ep_length),
                    "joint_reward": episode_return_true + episode_return_cost,
                }
                self.log_save_logs(train=False, logs=logs, episode=episode, save_logs=True)
                eval_count += 1

            ep_seed = cantor_pairing(self._rng_seed, episode)
            obs, _ = self._env.reset(seed=ep_seed)
            episode_return_true = 0.0
            episode_return_proxy = 0.0
            episode_return_cost = 0.0
            episode_loss_mdp = 0.0
            episode_loss_mon = 0.0
            episode_reward_model_loss = 0.0
            reward_seen = False
            episode_monitor_action_count, episode_timesteps = 0, 0
            next_action = None
            ep_joint_reward = []
            agent_pos, grid_size = (0, 0), (9, 9)  # TODO bug if (3, 3) fix this
            current_device = self._critic.get_device()
            while True:
                episode_timesteps += 1
                action = self._actor(obs) if next_action is None else next_action
                self._visit_table[
                    self.get_obs_from_agent_pos(self, grid_size, agent_pos), get_action_ind(action)
                ] += 1  # fix for stateMonMDP

                next_obs, reward, term, trunc, info = self._env.step(action)
                agent_pos = info["agent_pos"]
                grid_size = info["grid"].shape
                if self.buffer is not None:
                    self.buffer.push(obs, action, {"mdp": None, "monitor": None} if term else next_obs, reward)

                if action["monitor"] == 1:
                    episode_monitor_action_count += 1

                if self._critic._on_policy:
                    next_action = self._actor(next_obs)
                if self.buffer is None:
                    step_loss_mdp, step_loss_mon = self._critic.update(obs, action, reward, term, next_obs, next_action)
                else:
                    # optimize
                    step_loss_mdp, step_loss_mon = 0, 0

                episode_return_true += info["mdp_reward"]
                episode_return_cost += reward["monitor"]
                if not np.isnan(reward["mdp"]):
                    reward_seen = True
                    episode_return_proxy += reward["mdp"]
                if not np.isnan(step_loss_mdp):
                    episode_loss_mdp += step_loss_mdp
                if not np.isnan(step_loss_mon):
                    episode_loss_mon += step_loss_mon

                ep_joint_reward.append(info["mdp_reward"] + reward["monitor"])
                self._actor.update()
                if term or trunc:
                    if not reward_seen:
                        episode_return_proxy = np.nan
                    break

                obs = next_obs
            # Update Q-network and reward network
            if self.buffer.buffer_size > self._start_train_timestep:
                for epoch in range(n_epoches_per_timesteps):
                    batch = self.buffer.process_batch(self.buffer.sample(batch_size), device=current_device)
                    step_loss_mdp, r_model_loss = self._critic.optimize_policy_model(batch)
                    episode_reward_model_loss += r_model_loss
                    episode_loss_mdp += step_loss_mdp
                    episode_loss_mon += step_loss_mon
            joint_reward.update({episode: ep_joint_reward})
            total_timesteps += episode_timesteps
            logs = {
                "environment_reward": episode_return_true,
                "received_reward": episode_return_proxy,
                "monitor_reward": episode_return_cost,
                "loss_mdp": episode_loss_mdp,
                "loss_mon": episode_loss_mon,
                "episode_reward_model_loss": episode_reward_model_loss,
                "monitor_action": episode_monitor_action_count,
                "number_of_timesteps": episode_timesteps,
                "joint_reward": episode_return_true + episode_return_cost,
            }
            self.log_save_logs(train=True, logs=logs, episode=episode, save_logs=True)
            episode += 1

        # save Q-table as numpy array
        # if self.buffer is not None:
        #     self.buffer.save(log_dir=self._log_dir)
        self._critic.save(seed=self._rng_seed)
        if self._save_train_log:
            np.save(self._log_dir + "/visit_table_{}.npy".format(self._rng_seed), self._visit_table)
            np.save(self._log_dir + "/training_joint_reward_{}.npy".format(self._rng_seed), joint_reward)
            np.save(
                self._log_dir + "/evaluation_joint_reward_{}.npy".format(self._rng_seed),
                eval_joint_reward,
            )
        wandb.finish()
        self._env.close()

    def test(self, render: bool = False, seed: int = 1, save_results: bool = False):
        """Evaluate an algorithm in Monitor MDP env, logs and save results"""
        episode_return_true = []
        episode_return_proxy = []
        episode_return_cost = []
        episode_monitor_action = []
        episode_length = []
        episode_discount_reward = []
        trajectories = {}
        total_timesteps = 0
        episode = 0
        while episode < self._testing_episodes:
            reward_seen = False
            ep_seed = cantor_pairing(self._rng_seed, episode)
            obs, _ = self._env.reset(seed=ep_seed)
            episode_monitor_action_count, episode_timesteps = 0, 0
            ep_states, ep_actions, ep_joint_reward = [], [], []
            return_true, return_proxy, return_cost, ep_discount_reward = 0, 0, 0, 0
            while True:
                if isinstance(self._env.observation_space["mdp"], gym.spaces.Discrete):
                    ep_states.append([obs["mdp"].item(), obs["monitor"]])
                if render:
                    self._env.render()
                action = self._actor(obs)
                ep_actions.append([action["mdp"], action["monitor"]])
                if action["monitor"] == 1:
                    episode_monitor_action_count += 1
                next_obs, reward, term, trunc, info = self._env.step(action)
                return_true += info["mdp_reward"]
                return_cost += reward["monitor"]
                ep_joint_reward.append(info["mdp_reward"] + reward["monitor"])
                ep_discount_reward += (self._critic._gamma**episode_timesteps) * (
                    info["mdp_reward"] + reward["monitor"]
                )
                if not np.isnan(reward["mdp"]):
                    reward_seen = True
                    return_proxy += reward["mdp"]
                if term or trunc:
                    if not reward_seen:
                        return_proxy = np.nan
                    break
                obs = next_obs
                episode_timesteps += 1

            episode_return_true.append(return_true)
            episode_return_cost.append(return_cost)
            episode_return_proxy.append(return_proxy)
            episode_monitor_action.append(episode_monitor_action_count)
            episode_length.append(episode_timesteps)
            total_timesteps += episode_timesteps
            episode_discount_reward.append(ep_discount_reward)
            trajectories[episode] = {
                "states": np.array(ep_states),
                "actions": np.array(ep_actions),
                "environment_reward": np.array(episode_return_true),
                "received_reward": np.array(episode_return_proxy),
                "monitor_reward": np.array(episode_return_cost),
                "joint_reward": np.array(episode_return_true) + np.array(episode_return_cost),
                "length": np.array(episode_length),
                "undiscounted_joint_reward": np.array(ep_joint_reward),
            }
            episode += 1
        if save_results:
            np.save(self._log_dir + "/trajectories_{}.npy".format(seed), trajectories)
        return (
            np.array(episode_return_true),
            np.array(episode_return_proxy),
            np.array(episode_return_cost),
            np.array(episode_monitor_action),
            np.array(episode_length),
            np.array(episode_discount_reward),
            trajectories,
        )

    def checkpoint(self):
        """save the model and statistic during the training process"""
        checkpoint_dir = self._log_dir + "checkpoints_{}/".format(self._checkpoint_count)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._critic.save(file_name="checkpoints_{}/".format(self._checkpoint_count), seed=self._rng_seed)
        np.save(checkpoint_dir + "/visit_table_{}.npy".format(self._rng_seed), self._visit_table)
        # if self.buffer is not None:
        #     self.buffer.save(log_dir=checkpoint_dir)
        self._checkpoint_count += 1

    @staticmethod
    def get_obs_from_agent_pos(self, grid_size: (int, int), agent_pos: (int, int)) -> int:
        return int(agent_pos[0] * grid_size[0] + agent_pos[1])

    def log_save_logs(self, train: bool, logs: dict, episode: int, save_logs: bool) -> None:
        """log and save logs to wand"""
        for key, value in logs.items():
            wandb.log({"{}/{}".format("train" if train else "test", key): value}, step=episode, commit=True)
            if save_logs:
                np.save(self._log_dir + "/{}_{}.npy".format(key, self._rng_seed), value)
