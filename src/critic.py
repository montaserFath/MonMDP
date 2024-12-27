# pylint: disable=no-member, too-many-arguments, arguments-differ, super-init-not-called
"""Critic for MDP and Monitored MDP"""
from abc import ABC, abstractmethod
from collections import namedtuple
import os
import numpy as np
import torch
from src.reward import RTable, RDict, RewardNet
from src.network import CNN


class Critic(ABC):
    """Generic class for the critic"""

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @abstractmethod
    def update(self, **kwargs):
        """Update the critic"""
        return

    @abstractmethod
    def reset(self):
        """reset the critic"""
        return

    @abstractmethod
    def report(self):
        """get the current status of the critic"""
        return

    @abstractmethod
    def optimize_policy_model(self, batch: dict, update_target: bool = False, monitor_state: bool = False, rewards = None):
        return


# ------------------------------------------------------------------------------
# Classic MDP
# ------------------------------------------------------------------------------


class QCritic(Critic):
    """Critic for Q-learning (as well as SARSA) in MDP"""

    def __init__(self, q0: float = 0.0, gamma=0.99, lr: float = 0.01, on_policy: bool = False, **kwargs):
        self._q0 = q0
        self._gamma = gamma
        self._lr = lr
        self._on_policy = on_policy

    def update(self, state, action, reward, terminated, next_state, next_action=None):
        if self._on_policy:  # TODO: on policy is not working fix it
            q_next = self(next_state, next_action)
        else:
            q_next = self(next_state).max()
        target = reward + self._gamma * (1.0 - terminated) * q_next
        prediction = self(state, action)
        new_value = (1.0 - self._lr) * prediction + self._lr * target
        self._update(state, action, new_value)
        return 0.5 * (target - prediction) ** 2

    @property
    def n_actions(self):
        """get the number of the environment actions"""
        return self._n_actions


class QTable(QCritic):
    """Q-Table for critic in MDP"""

    def __init__(
            self,
            observation_space,
            action_space,
            dir_name: str = None,
            q0: float = 0.0,
            gamma: float = 0.99,
            lr: float = 0.01,
            on_policy: bool = False,
            env_name: str = None,
            **kwargs,
    ):
        QCritic.__init__(self, q0, gamma, lr, on_policy)
        self._n_states = observation_space.n
        self._n_actions = action_space.n
        self._q_table = None
        if env_name is not None:
            self.env_size = "3_3" if env_name.split("/")[1].split("-")[-1] == "v0" else "9_9"  # TODO change this
            self._dir_name = (
                "models/{}/{}/q_learning/".format(self.env_size, env_name.split("-")[1])
                if dir_name is None
                else dir_name
            )
        self.reset()

    def __call__(self, state, action=None):
        state = state.item() if isinstance(state, np.ndarray) else state
        if action is None:
            return self._q_table[state]
        return self._q_table[state][action]

    def _update(self, state, action, new_value):
        state = state.item() if isinstance(state, np.ndarray) else state
        self._q_table[state][action] = new_value

    def reset(self):
        """reset the values of the q_table to their initial values"""
        shp = (self._n_states, self._n_actions)
        self._q_table = np.ones(shp) * self._q0

    def report(self):
        """get the current q_table values"""
        return self._q_table

    def save(self, seed: int = 1, file_name: str = None):
        """save q-table as a numpy array"""
        file_dir = self._dir_name if file_name is None else self._dir_name + "/" + file_name
        os.makedirs(file_dir, exist_ok=True)
        np.save(file_dir + "/critic_q_table_{}.npy".format(seed), self._q_table)

    def load(self, log_dir: str = None, seed: int = 1, file_name: str = None):
        """load the q-table which saved as a numpy array"""
        if log_dir is None:
            raise ValueError("No files to load Q-Table from it")
        file_dir = log_dir if file_name is None else log_dir + "/" + file_name
        self._q_table = np.load(file_dir + "/critic_q_table_{}.npy".format(seed))


class QDict(QCritic):
    """Dictionary Q for the critic in MDP"""

    def __init__(self, observation_space, action_space, q0=0.0, gamma=0.99, lr=0.01, on_policy=False, **kwargs):
        QCritic.__init__(self, q0, gamma, lr, on_policy)
        self._n_actions = action_space.n
        self._q_dict = None
        self.reset()

    def __call__(self, state, action=None):
        if action is None:
            return np.array([q.get(tuple(state), self._q0) for q in self._q_dict])
        else:
            return self._q_dict[action].get(tuple(state), self._q0)

    def _update(self, state, action, new_value):
        self._q_dict[action][tuple(state)] = new_value

    def reset(self):
        """reset the values of the q-dictionary to their initial values"""
        self._q_dict = [dict() for _ in range(self._n_actions)]

    def report(self):
        """get the current values of the q-dictionary"""
        return self._q_dict


# ------------------------------------------------------------------------------
# Monitored MDP
# ------------------------------------------------------------------------------


# pylint: disable=too-many-instance-attributes, too-many-locals, too-many-branches
class MonQCritic(Critic):
    """Dictionary Q for the critic in Monitored MDP"""

    def __init__(
            self,
            env_name: str,
            q0=0.0,
            gamma=0.99,
            lr=0.01,
            on_policy=False,
            strategy: str = "reward_model",
            unseen_r_value: float = 0.0,
            **kwargs,
    ):
        self._env_name = env_name
        self._q0 = q0
        self._gamma = gamma
        self._lr = lr
        self._on_policy = on_policy
        self._strategy = strategy
        self._unseen_r_value = unseen_r_value
        self._mdp_q = None
        self._mon_q = None
        self._r_model = None
        self._mdp_critic = None

    def update(self, state, action, reward, terminated, next_state, next_action=None):
        """Update the q-value"""
        if not np.isnan(reward["mdp"]):
            if self._strategy == "reward_model":
                self._r_model.update(state["mdp"], action["mdp"], reward["mdp"])
        else:
            if self._strategy == "reward_model":
                reward["mdp"] = self._r_model(state["mdp"], action["mdp"])
            elif self._strategy == "ignore":
                return np.nan, np.nan
            elif self._strategy == "zero_reward":
                reward["mdp"] = self._unseen_r_value
            elif self._strategy in ["q_mdp", "q_monitor_sequential", "q_monitor_joint"]:
                return np.nan, np.nan
            else:
                raise ValueError("unknown update strategy")

        if not np.isnan(reward["mdp"]):
            mdp_error = self._mdp_critic.update(
                state["mdp"], action["mdp"], reward["mdp"], terminated, next_state["mdp"]
            )
        else:
            mdp_error = np.nan

        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            # update Q for MDP and MonMDP
            q_next = self(next_state, next_action if self._on_policy else None)
            prediction = self(state, action)
            new_value, error = {}, {}
            for key in q_next.keys():
                q_next_value = q_next[key].max() if isinstance(q_next[key], np.ndarray) else q_next[key].item()
                target = reward[key] + self._gamma * (1.0 - terminated) * q_next_value
                new_value[key] = (1.0 - self._lr) * prediction[key] + self._lr * target
                error[key] = 0.5 * (target - prediction[key]) ** 2
            self._update(state, action, new_value)
            return error["mdp"], error["monitor"]

        if not np.isnan(reward["mdp"]):
            joint_reward = reward["monitor"] + reward["mdp"]
        else:
            joint_reward = reward["monitor"] + 0.0

        if self._on_policy:
            q_next = self(next_state, next_action).item()
        else:
            q_next = self(next_state).max()
        target = joint_reward + self._gamma * (1.0 - terminated) * q_next
        prediction = self(state, action)
        new_value = (1.0 - self._lr) * prediction + self._lr * target
        self._update(state, action, new_value)
        mon_error = 0.5 * (target - prediction) ** 2

        return mdp_error, mon_error

    @property
    def n_actions(self):
        """get the number of environment actions"""
        return self._n_actions

    @property
    def n_mon_actions(self):
        """get th number of the monitored actions"""
        return self._n_mon_actions


class MonQTable(MonQCritic):
    """Q-Table for critic in Monitored MDP"""

    def __init__(
            self,
            env_name,
            observation_space,
            action_space,
            dir_name: str = None,
            q0=0.0,
            gamma=0.99,
            lr=0.01,
            on_policy=False,
            strategy="zero_reward",
            unseen_r_value=0.0,
            **kwargs,
    ):
        MonQCritic.__init__(self, env_name, q0, gamma, lr, on_policy, strategy=strategy, unseen_r_value=unseen_r_value)
        self._mdp_critic = QTable(observation_space["mdp"], action_space["mdp"], q0, gamma, lr)
        self._n_states = observation_space["mdp"].n
        self._n_actions = action_space["mdp"].n
        self._n_mon_states = observation_space["monitor"].n
        self._n_mon_actions = action_space["monitor"].n
        self._q_table = None
        self.env_size = "3_3" if env_name.split("/")[1].split("-")[-1] == "v0" else "9_9"  # TODO change this
        env_name = self._env_name.split("/")[1].split("-")[1]
        self._dir_name = (
            "models/{}/{}/{}/".format(self.env_size, env_name, self._strategy) if dir_name is None else dir_name
        )

        if self._strategy == "reward_model":
            self._r_model = RTable(observation_space["mdp"], action_space["mdp"], **kwargs["reward_model"])

        self.reset()

    def __call__(self, state, action=None):
        if action is None:
            return self._q_table[state["mdp"]][state["monitor"]]
        return self._q_table[state["mdp"]][state["monitor"]][action["mdp"]][action["monitor"]]

    def _update(self, state, action, new_value):
        self._q_table[state["mdp"]][state["monitor"]][action["mdp"]][action["monitor"]] = new_value

    def reset(self):
        """reset the values of the q-table to initial values in Monitored MDP"""
        shp = (self._n_states, self._n_mon_states, self._n_actions, self._n_mon_actions)
        self._q_table = np.ones(shp) * self._q0
        self._mdp_critic.reset()

    def report(self):
        """get the values of the q-table in monitored MDP"""
        return self._q_table, self._mdp_critic.report()


class MonQNet(MonQCritic):
    def __init__(
            self,
            env_name: str,
            observation_space,
            action_space,
            kernel_size_0: int,
            kernel_size_1: int,
            stride_0: int,
            stride_1: int,
            device: str,
            q0=0.0,
            gamma=0.99,
            lr=0.01,
            on_policy=False,
            strategy: str = "reward_model",
            unseen_r_value: float = 0.0,
            dir_name: str = None,
            **kwargs,
    ):
        MonQCritic.__init__(self, env_name, q0, gamma, lr, on_policy, strategy=strategy, unseen_r_value=unseen_r_value)
        self._n_actions = action_space["mdp"].n
        self._n_mon_actions = action_space["monitor"].n
        self._q0 = q0
        self._gamma = gamma
        self._lr = lr
        self._kernel_size_0 = kernel_size_0
        self._kernel_size_1 = kernel_size_1
        self._stride_0 = stride_0
        self._stride_1 = stride_1
        self._device = device
        # self.replay_buffer = None
        self._q_network = None
        self._loss_fun = torch.nn.MSELoss()
        self.optimizer = None
        self._transition = None
        self._strategy = strategy
        env_size = "3_3" if env_name.split("/")[1].split("-")[-1] == "v0" else "10_10"  # TODO change this
        self._dir_name = "models/{}/{}/{}/".format(env_size, env_name, self._strategy) if dir_name is None else dir_name
        self._observation_space = observation_space
        self._action_space = action_space
        self.reset()

    def reset(self):
        self._q_network.reset()

    def report(self):
        NotImplemented

    def _update(self, state, action, new_value):
        NotImplemented

    def __call__(self, state, action=None):
        mdp_state = state["mdp"]
        state = mdp_state if torch.is_tensor(mdp_state) else torch.tensor(mdp_state, dtype=torch.float)
        state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2]) if len(state.shape) == 3 else state
        q_state = self._q_network(state).detach().numpy()
        if action is None:
            return q_state
        return q_state[action]


class MonQCNN(MonQNet):
    def __init__(
            self,
            env_name: str,
            observation_space,
            action_space,
            kernel_size_0: int,
            kernel_size_1: int,
            stride_0: int,
            stride_1: int,
            device: str,
            **kwargs,
    ):
        super().__init__(
            env_name, observation_space, action_space, kernel_size_0, kernel_size_1, stride_0, stride_1, device, **kwargs,
        )
        self._kernel_size_0 = kernel_size_0
        self._kernel_size_1 = kernel_size_1
        self._stride_0 = stride_0
        self._stride_1 = stride_1
        self._device = device
        self._target_network = None
        self._tau = 5e-4
        if self._lr > 1e-2:
            raise ValueError("Learning rate for NN should be small not {}".format(self._lr))
        self.optimizer = torch.optim.Adam(self._q_network.model.parameters(), lr=self._lr)
        self.q_loss_fun = torch.nn.MSELoss()
        self._q_net_loss = []
        if self._strategy == "reward_model":
            self._r_model = RewardNet(observation_space["mdp"], action_space["mdp"], **kwargs["reward_model"])
            self._r_model.reset()

    def reset(self):
        n_actions = int(self._action_space["mdp"].n * self._action_space["monitor"].n)
        self._q_network = CNN(
            self._observation_space["mdp"].shape,
            n_actions,
            self._lr,
            self._kernel_size_0,
            self._kernel_size_1,
            self._stride_0,
            self._stride_1,
            device=self._device,
        )
        self._target_network = CNN(
            self._observation_space["mdp"].shape,
            n_actions,
            self._lr,
            self._kernel_size_0,
            self._kernel_size_1,
            self._stride_0,
            self._stride_1,
            device=self._device,
        )
        self._transition = namedtuple("Transition", ("obs", "action", "next_obs", "reward"))
        self._q_network.init_network()
        self._target_network.init_network()
        self.optimizer = torch.optim.Adam(self._q_network.model.parameters(), lr=self._lr)
        self._q_net_loss = []

    def __call__(self, state, action=None):
        q_state = self._q_network.forward(state["mdp"]).detach().cpu().numpy()
        if action is None:
            return q_state
        return q_state[action]

    # optimize the Q-network once
    def optimize_policy_model(self, batch: dict, update_target: bool = False, monitor_state: bool = False, rewards = None):
        real_idx = batch["real_reward_idx"]
        if rewards is None:
            if self._strategy == "reward_model":
                reward_loss = self._r_model.optimize_reward_model(batch)
                with torch.no_grad():
                    # mdp_rewards = self._r_model._network(batch["mdp_obs"]).gather(1, batch["mdp_action"])
                    mdp_rewards = self._r_model(batch["mdp_obs"]).gather(1, batch["mdp_action"])
            elif self._strategy == "zero_reward":
                mdp_rewards = self._unseen_r_value * torch.ones_like(batch["mdp_reward"], dtype=torch.float, device=self._device)
                mdp_rewards[real_idx] = batch["mdp_reward"][real_idx]
                reward_loss = 0
            elif self._strategy == "ignore":
                reward_loss = 0
                mdp_rewards = batch["mdp_reward"][real_idx]
            else:
                raise NotImplementedError
        else:
            mdp_rewards = rewards.gather(1, batch["mdp_action"])
            reward_loss = 0
        mon_rewards = batch["mon_reward"][real_idx] if self._strategy == "ignore" else batch["mon_reward"]
        combined_reward = mdp_rewards + mon_rewards
        if self._strategy == "ignore":
            combined_action = batch["mdp_action"][real_idx] + self._action_space["mdp"].n * batch["mon_action"][real_idx]
        else:
            combined_action = batch["mdp_action"] + self._action_space["mdp"].n * batch["mon_action"]

        mdp_obs = batch["mdp_obs"][real_idx] if self._strategy == "ignore" else batch["mdp_obs"]
        mon_obs = batch["mon_obs"][real_idx] if self._strategy == "ignore" else batch["mon_obs"]
        q_values = self._q_network.forward(mdp_obs, mon_obs if monitor_state else None).gather(1, combined_action)
        next_q_values = torch.zeros(mdp_obs.shape[0], device=self._device)

        non_final = batch["non_final_mask"][real_idx] if self._strategy == "ignore" else batch["non_final_mask"]
        non_mdp = batch["non_final_next_states"][real_idx] if self._strategy == "ignore" else batch["non_final_next_states"]
        non_mon = batch["non_final_next_monitor_states"][real_idx] if self._strategy == "ignore" else batch["non_final_next_monitor_states"]
        with torch.no_grad():
            next_q_values[non_final] = (
                self._target_network.forward(non_mdp, non_mon if monitor_state else None).max(1).values
            )
        expected_q_values = (self._gamma * next_q_values.unsqueeze(1)) + combined_reward
        q_loss = self.q_loss_fun(q_values, expected_q_values)

        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()
        self._q_net_loss.append(q_loss.item())

        # update the target network
        if update_target:
            target_net_state_dict = self._target_network.model.state_dict()
            policy_net_state_dict = self._q_network.model.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self._tau + target_net_state_dict[key] * (
                        1 - self._tau
                )
            self._target_network.model.load_state_dict(target_net_state_dict)

        return q_loss.item(), reward_loss

    def save(self, seed: int = 1, file_name: str = None):
        file_dir = self._dir_name if file_name is None else self._dir_name + "/" + file_name
        os.makedirs(file_dir, exist_ok=True)
        self._q_network.save(log_dir=file_dir + "/q_network_{}".format(seed))
        # self._target_network.save(log_dir=file_dir + "/target_network_{}".format(seed))
        if self._strategy == "reward_model":
            self._r_model.save(seed=seed, file_name=file_dir)

        np.save(file_dir + "/q_network_loss_{}".format(seed), self.get_current_loss())

    def load(self, seed: int = 1, file_name: str = None):
        file_dir = self._dir_name if file_name is None else self._dir_name + "/" + file_name
        self._q_network.load(log_dir=file_dir + "/q_network_{}".format(seed))
        self._target_network.load(log_dir=file_dir + "/target_network_{}".format(seed))
        self._r_model.load(seed=seed, file_name=file_dir)

    def update(self, state, action, reward, terminated, next_state, next_action=None):
        if not np.isnan(reward["mdp"]):
            if self._strategy == "reward_model":
                self._r_model.update(state["mdp"], action["mdp"], reward["mdp"])
        else:
            if self._strategy == "zero_reward":
                reward["mdp"] = self._unseen_r_value
            elif self._strategy == "ignore":
                return np.nan, np.nan
            else:
                raise NotImplementedError


    def report(self):
        NotImplemented

    def get_device(self) -> str:
        return self._device

    def get_current_loss(self):
        return self._q_net_loss


class MonRoomCNN(MonQCNN):
    def __init__(
            self, env_name, observation_space, action_space, kernel_size_0, kernel_size_1, stride_0, stride_1, device, **kwargs,
    ):
        super().__init__(
            env_name, observation_space, action_space, kernel_size_0, kernel_size_1, stride_0, stride_1, device, **kwargs,
        )

    def reset(self):
        self._q_network = CNN(
            self._observation_space["mdp"].shape,
            self._action_space["mdp"].n,
            self._lr,
            self._kernel_size_0,
            self._kernel_size_1,
            self._stride_0,
            self._stride_1,
            add_monitor_obs=True,
            device=self._device,
        )
        self._target_network = CNN(
            self._observation_space["mdp"].shape,
            self._action_space["mdp"].n,
            self._lr,
            self._kernel_size_0,
            self._kernel_size_1,
            self._stride_0,
            self._stride_1,
            add_monitor_obs=True,
            device=self._device,
        )
        self._transition = namedtuple("Transition", ("obs", "action", "next_obs", "reward"))
        self._q_network.init_network()
        self._target_network.init_network()
        self.optimizer = torch.optim.Adam(self._q_network.model.parameters(), lr=self._lr)
        self._q_net_loss = []

    def __call__(self, state, action=None):
        q_state = self._q_network.forward(state["mdp"], state["monitor"]).detach().cpu().numpy()
        if action is None:
            return q_state
        return q_state[action]

    def optimize_policy_model(self, batch: dict, update_target: bool = False, monitor_state: bool = False, rewards = None):
        return super().optimize_policy_model(batch, update_target, monitor_state, rewards)

    def save(self, seed: int = 1, file_name: str = None):
        super().save(seed=seed, file_name=file_name)

    def update(self, state, action, reward, terminated, next_state, next_action=None):
        super().update(state, action, reward, terminated, next_state, next_action)

    def load(self, seed: int = 1, file_name: str = None):
        super().load(seed=seed, file_name=file_name)

    def get_current_loss(self):
        return self._q_net_loss

    def get_device(self):
        return self._device


class MonQTableOneAction(MonQTable):
    """Q-table for Monitored MDP with a single monitoring action"""

    def __int__(self, env_name, observation_space, action_space, **kwargs):
        super().__int__(self, env_name, observation_space, action_space)

    def reset(self):
        """reset the values of the q-table to initial values in Monitored MDP"""
        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            self._mdp_q = np.ones((self._n_states, self._n_actions)) * self._q0
            self._mon_q = np.ones((self._n_states, self._n_actions * self._n_mon_actions)) * 0  # TODO: remove 0
        else:
            self._q_table = np.ones((self._n_states, self._n_actions * self._n_mon_actions)) * self._q0
        self._mdp_critic.reset()

    def __call__(self, state, action=None):
        mdp_s = state["mdp"]
        if action is None:
            if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
                mdp_q = self.expand_mdp_q()[mdp_s] if self._strategy == "q_monitor_joint" else self._mdp_q[mdp_s]
                return {"mdp": mdp_q, "monitor": self._mon_q[mdp_s]}
            return self._q_table[mdp_s]

        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            mdp_q = self.expand_mdp_q()[mdp_s] if self._strategy == "q_monitor_joint" else self._mdp_q[mdp_s]
            mon_q = self._mon_q[mdp_s, self.get_action_ind(action)]
            return {"mdp": np.squeeze(mdp_q)[action["mdp"]], "monitor": mon_q}
        return self._q_table[mdp_s, self.get_action_ind(action)]

    def _update(self, state, action, new_value):
        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            self._mdp_q[state["mdp"], action["mdp"]] = new_value["mdp"]
            self._mon_q[state["mdp"], self.get_action_ind(action)] = new_value["monitor"]
        else:
            self._q_table[state["mdp"], self.get_action_ind(action)] = new_value

    def get_action_ind(self, action: dict) -> int:
        """convert a dictionary of MDP and Monitor actions to an index"""
        mdp_action, mon_action = action["mdp"], action["monitor"]
        return mon_action * self._n_actions + mdp_action

    def ind_to_action(self, action_ind: int) -> dict:
        """convert the action index to a dictionary of MDP and Monitor actions"""
        mon_action, mdp_action = action_ind // self._n_actions, action_ind % self._n_actions
        return {"mdp": mdp_action, "monitor": mon_action}

    def save(self, seed: int = 1, file_name: str = None):
        """save the q-table as numpy array"""
        file_dir = self._dir_name if file_name is None else self._dir_name + "/" + file_name
        os.makedirs(file_dir, exist_ok=True)
        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            np.save(file_dir + "/mdp_q_table_{}.npy".format(seed), self._mdp_q)
            np.save(file_dir + "/monitor_q_table_{}.npy".format(seed), self._mon_q)
        else:
            np.save(file_dir + "/critic_q_table_{}.npy".format(seed), self._q_table)
        if self._r_model is not None:
            self._r_model.save(file_dir, seed)

    def load(self, log_dir: str = None, seed: int = 1, file_name: str = None) -> None:
        """Load a q-table which saved as a numpy array"""
        if log_dir is None:
            raise ValueError("No files to load Q-Table from it")
        file_dir = log_dir if file_name is None else log_dir + "/" + file_name
        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            self._mdp_q = np.load(file_dir + "/mdp_q_table_{}.npy".format(seed))
            self._mon_q = np.load(file_dir + "/monitor_q_table_{}.npy".format(seed))
        else:
            self._q_table = np.load(file_dir + "/critic_q_table_{}.npy".format(seed))
        if self._r_model is not None:
            self._r_model = np.load(file_dir + "/reward_model_table_{}.npy".format(seed))

    def expand_mdp_q(self):
        """Expand Q-table for MDP"""
        new_q = np.zeros((self._mdp_q.shape[0], self._mdp_q.shape[1] * 2))
        new_q[:, : self._n_actions] = self._mdp_q.copy()
        new_q[:, self._n_actions:] = self._mdp_q.copy()
        return new_q


class StateMonTable(MonQTableOneAction):
    """Q-table for Monitored MDP with a monitoring state"""

    def reset(self):
        """reset the q-table in a state monitor MDP to initial value"""
        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            self._mdp_q = np.ones((self._n_states, self._n_actions)) * self._q0
            self._mon_q = np.ones((self._n_states * self._n_mon_states, self._n_actions)) * 0  # TODO: remove 0
        else:
            self._q_table = np.ones((self._n_states * self._n_mon_states, self._n_actions)) * self._q0
        self._mdp_critic.reset()

    def __call__(self, state, action=None):
        mdp_state, mdp_action = state["mdp"].item(), action["mdp"] if action is not None else None
        state_ind = self.get_state_ind(state)

        if action is None:
            if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
                mdp_q = (
                    self.expand_mdp_q()[state_ind] if self._strategy == "q_monitor_joint" else self._mdp_q[mdp_state]
                )
                return {"mdp": mdp_q, "monitor": self._mon_q[state_ind]}
            return self._q_table[state_ind]

        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            mdp_q = self.expand_mdp_q()[state_ind] if self._strategy == "q_monitor_joint" else self._mdp_q[mdp_state]
            mon_q = self._mon_q[state_ind, mdp_action]
            return {"mdp": np.squeeze(mdp_q)[mdp_action], "monitor": mon_q}
        return self._q_table[self.get_state_ind(state), mdp_action]

    def _update(self, state, action, new_value):
        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            self._mdp_q[state["mdp"], action["mdp"]] = new_value["mdp"]
            self._mon_q[self.get_state_ind(state), action["mdp"]] = new_value["monitor"]
        else:
            self._q_table[self.get_state_ind(state), action["mdp"]] = new_value

    def get_state_ind(self, state: dict) -> int:
        """convert MDP and Monitor states to an index"""
        return state["monitor"] * self._n_states + state["mdp"].item()

    def ind_to_state(self, state_ind: int) -> dict:
        """convert state index to a dictionary of MDP and Monitor states"""
        if state_ind >= self._n_states * self._n_mon_states:
            raise ValueError("State index is larger than max Number of states")
        mon_state, mdp_state = state_ind // self._n_states, state_ind % self._n_states
        return {"mdp": mdp_state, "monitor": mon_state}

    def expand_mdp_q(self):
        """Expand Q-table for MDP"""
        new_q = np.zeros((self._mdp_q.shape[0] * 2, self._mdp_q.shape[1]))
        new_q[: self._n_states] = self._mdp_q.copy()
        new_q[self._n_states:] = self._mdp_q.copy()
        return new_q


# pylint: disable=too-many-instance-attributes
class MonQDict(MonQCritic):
    """Q-Dictionary for Monitored MDP"""

    def __init__(
            self,
            observation_space,
            action_space,
            q0=0.0,
            gamma=0.99,
            lr=0.01,
            on_policy=False,
            strategy="zero_reward",
            **kwargs,
    ):
        MonQCritic.__init__(self, q0, gamma, lr, on_policy)
        self._mdp_critic = QDict(observation_space["mdp"], action_space["mdp"], q0, gamma, lr)
        self._n_actions = action_space["mdp"].n
        self._n_mon_actions = action_space["monitor"].n
        self._q0 = q0
        self._gamma = gamma
        self._lr = lr
        self._q_dict = None

        if self._strategy == "reward_model":
            self._r_model = RDict(observation_space["mdp"], action_space["mdp"], **kwargs["reward_model"])
        else:
            self._r_model = None

        self.reset()

    def __call__(self, state, action=None):
        state_full = np.concatenate((state["mdp"], [state["monitor"]]))

        if action is None:
            return np.array([[q_mon.get(tuple(state_full), self._q0) for q_mon in q_mdp] for q_mdp in self._q_dict])
        return self._q_dict[action["mdp"]][action["monitor"]].get(tuple(state), self._q0)

    def _update(self, state, action, new_value):
        state = np.concatenate((state["mdp"], [state["monitor"]]))
        self._q_dict[action["mdp"]][action["monitor"]][tuple(state)] = new_value

    def reset(self):
        """reseat the Q-dictionary to initial value"""
        self._q_dict = [[dict() for _ in range(self._n_mon_actions)] for _ in range(self._n_actions)]
        self._mdp_critic.reset()

    def report(self):
        """get the current values of Q-dictionary"""
        return self._q_dict, self._mdp_critic.report()
