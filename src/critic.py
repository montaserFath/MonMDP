# import datetime
import os
import numpy as np
from abc import ABC, abstractmethod
from src.reward import RTable, RDict


class Critic(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @abstractmethod
    def update(self, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def report(self):
        pass


# ------------------------------------------------------------------------------
# Classic MDP
# ------------------------------------------------------------------------------

class QCritic(Critic):
    def __init__(self, q0=0., gamma=0.99, lr=0.01, on_policy=False, **kwargs):
        self._q0 = q0
        self._gamma = gamma
        self._lr = lr
        self._on_policy = on_policy

    def update(self, state, action, reward, terminated, next_state, next_action=None):
        if self._on_policy:  # TODO(Monta): on policy is not working fi it
            q_next = self(next_state, next_action)
        else:
            q_next = self(next_state).max()
        target = reward + self._gamma * (1. - terminated) * q_next
        prediction = self(state, action)
        new_value = (1. - self._lr) * prediction + self._lr * target
        self._update(state, action, new_value)
        return 0.5 * (target - prediction) ** 2

    @property
    def n_actions(self):
        return self._n_actions


class QTable(QCritic):
    def __init__(self, observation_space, action_space,
                 q0=0., gamma=0.99, lr=0.01, on_policy=False, **kwargs):
        QCritic.__init__(self, q0, gamma, lr, on_policy)
        self._n_states = observation_space.n
        self._n_actions = action_space.n
        self.reset()

    def __call__(self, state, action=None):
        state = state.item() if isinstance(state, np.ndarray) else state
        if action is None:
            return self._q_table[state]
        else:
            return self._q_table[state][action]

    def _update(self, state, action, new_value):
        state = state.item() if isinstance(state, np.ndarray) else state
        self._q_table[state][action] = new_value

    def reset(self):
        shp = (self._n_states, self._n_actions)
        self._q_table = np.ones(shp) * self._q0

    def report(self):
        return self._q_table


class QDict(QCritic):
    def __init__(self, observation_space, action_space,
                 q0=0., gamma=0.99, lr=0.01, on_policy=False, **kwargs):
        QCritic.__init__(self, q0, gamma, lr, on_policy)
        self._n_actions = action_space.n
        self.reset()

    def __call__(self, state, action=None):
        if action is None:
            return np.array([
                q.get(tuple(state), self._q0) for q in self._q_dict
            ])
        else:
            return self._q_dict[action].get(tuple(state), self._q0)

    def _update(self, state, action, new_value):
        self._q_dict[action][tuple(state)] = new_value

    def reset(self):
        self._q_dict = [dict() for _ in range(self._n_actions)]

    def report(self):
        return self._q_dict


# ------------------------------------------------------------------------------
# Monitored MDP
# ------------------------------------------------------------------------------

class MonQCritic(Critic):
    def __init__(
            self, q0=0., gamma=0.99, lr=0.01, on_policy=False, strategy: str = "reward_model", unseen_r_value: float = 0., **kwargs
    ):
        # TODO(Monta): fix load unseen_r_value from yamil file
        self._q0 = q0
        self._gamma = gamma
        self._lr = lr
        self._on_policy = on_policy
        self._strategy = strategy
        self._unseen_r_value = unseen_r_value

    def update(self, state, action, reward, terminated, next_state, next_action=None):
        if not np.isnan(reward['mdp']):
            if self._strategy == "reward_model":
                self._r_model.update(state['mdp'], action['mdp'], reward['mdp'])
        else:
            if self._strategy == "reward_model":
                reward['mdp'] = self._r_model(state['mdp'], action['mdp'])
            elif self._strategy == "ignore":
                return np.nan, np.nan
            elif self._strategy == "zero_reward":
                reward['mdp'] = self._unseen_r_value
            elif self._strategy in ["q_mdp", "q_monitor_sequential", "q_monitor_joint"]:
                return np.nan, np.nan
            else:
                raise ValueError('unknown update strategy')

        if not np.isnan(reward['mdp']):
            mdp_error = self._mdp_critic.update(
                state['mdp'], action['mdp'], reward['mdp'], terminated, next_state['mdp']
            )
        else:
            mdp_error = np.nan

        if not np.isnan(reward['mdp']):
            reward = reward['monitor'] + reward['mdp']
        else:
            reward = reward['monitor'] + 0.

        if self._on_policy:
            q_next = self(next_state, next_action).item()
        else:
            q_next = self(next_state).max()
        target = reward + self._gamma * (1. - terminated) * q_next
        prediction = self(state, action)
        new_value = (1. - self._lr) * prediction + self._lr * target
        self._update(state, action, new_value)
        mon_error = 0.5 * (target - prediction) ** 2

        return mdp_error, mon_error

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def n_mon_actions(self):
        return self._n_mon_actions


class MonQTable(MonQCritic):
    def __init__(self, observation_space, action_space,
                 q0=0., gamma=0.99, lr=0.01, on_policy=False,
                 strategy="zero_reward", unseen_r_value=0.0,
                 **kwargs):
        MonQCritic.__init__(self, q0, gamma, lr, on_policy, strategy=strategy, unseen_r_value=unseen_r_value)
        self._mdp_critic = QTable(
            observation_space['mdp'],
            action_space['mdp'],
            q0, gamma, lr
        )
        self._n_states = observation_space['mdp'].n
        self._n_actions = action_space['mdp'].n
        self._n_mon_states = observation_space['monitor'].n
        self._n_mon_actions = action_space['monitor'].n
        # env_name = env.spec.id.split("/")[1].split("-")[1]
        self._dir_name = "models/Simple/{}".format(self._strategy)
        # self._dir_name = "models/Simple/{}/{date:%Y_%m_%d_%H_%M_%S}".format(
        #     self._strategy, date=datetime.datetime.now()
        # )

        if self._strategy == "reward_model":
            self._r_model = RTable(
                observation_space['mdp'],
                action_space['mdp'],
                **kwargs['reward_model']
            )
        else:
            self._r_model = None

        self.reset()

    def __call__(self, state, action=None):
        if action is None:
            return self._q_table[state['mdp']][state['monitor']]
        else:
            return self._q_table[state['mdp']][state['monitor']][action['mdp']][action['monitor']]

    def _update(self, state, action, new_value):
        self._q_table[state['mdp']][state['monitor']][action['mdp']][action['monitor']] = new_value

    def reset(self):
        shp = (self._n_states, self._n_mon_states, self._n_actions, self._n_mon_actions)
        self._q_table = np.ones(shp) * self._q0
        self._mdp_critic.reset()

    def report(self):
        return self._q_table, self._mdp_critic.report()


class MonQTableOneAction(MonQTable):
    def __int__(self, observation_space, action_space, **kwargs):
        super().__int__(self, observation_space, action_space)

    def reset(self):
        if self._strategy == "q_mdp":
            table_shape = (self._n_states, self._n_actions)
        elif self._strategy == "q_monitor_sequential":
            raise NotImplemented
        elif self._strategy == "q_monitor_joint":
            raise NotImplemented
        else:
            table_shape = (self._n_states, self._n_actions * self._n_mon_actions)
        self._q_table = np.ones(table_shape) * self._q0
        self._mdp_critic.reset()

    def __call__(self, state, action=None):
        if action is None:
            return self._q_table[state["mdp"]]
        if self._strategy == "q_mdp":
            return self._q_table[state["mdp"], action["mdp"]]
        return self._q_table[state["mdp"], self.get_action_ind(action)]

    def _update(self, state, action, new_value):
        if self._strategy == "q_mdp":
            self._q_table[state["mdp"], action["mdp"]] = new_value
        else:
            self._q_table[state["mdp"], self.get_action_ind(action)] = new_value

    def get_action_ind(self, action: dict) -> int:
        mdp_action, mon_action = action["mdp"], action["monitor"]
        return mon_action * self._n_actions + mdp_action

    def ind_to_action(self, action_ind: int) -> dict:
        mon_action, mdp_action = action_ind // self._n_actions, action_ind % self._n_actions
        return {"mdp": mdp_action, "monitor": mon_action}

    def save(self):
        os.makedirs(self._dir_name, exist_ok=True)
        np.save(self._dir_name + "/critic_q_table.npy", self._q_table)
        if self._r_model is not None:
            self._r_model.save(self._dir_name)

    def load(self, log_dir: str = None):
        if log_dir is None:
            raise ValueError("No files to load Q-Table from it")
        self._q_table = np.load(log_dir + "/critic_q_table.npy")
        if self._r_model is not None:
            self._r_model = np.load(log_dir + "/reward_model_table.npy")


class MonQDict(MonQCritic):
    def __init__(self, observation_space, action_space,
                 q0=0., gamma=0.99, lr=0.01, on_policy=False,
                 strategy="zero_reward",
                 **kwargs):
        MonQCritic.__init__(self, q0, gamma, lr, on_policy)
        self._mdp_critic = QDict(
            observation_space['mdp'],
            action_space['mdp'],
            q0, gamma, lr
        )
        self._n_actions = action_space['mdp'].n
        self._n_mon_actions = action_space['monitor'].n
        self._q0 = q0
        self._gamma = gamma
        self._lr = lr

        if self._strategy == "reward_model":
            self._r_model = RDict(
                observation_space['mdp'],
                action_space['mdp'],
                **kwargs['reward_model']
            )
        else:
            self._r_model = None

        self.reset()

    def __call__(self, state, action=None):
        state_full = np.concatenate((state['mdp'], [state['monitor']]))

        if action is None:
            return np.array([
                [q_mon.get(tuple(state_full), self._q0) for q_mon in q_mdp]
                for q_mdp in self._q_dict
            ])
        else:
            return self._q_dict[action['mdp']][action['monitor']].get(
                tuple(state),
                self._q0
            )

    def _update(self, state, action, new_value):
        state = np.concatenate((state['mdp'], [state['monitor']]))
        self._q_dict[action['mdp']][action['monitor']][tuple(state)] = new_value

    def reset(self):
        self._q_dict = [
            [dict() for _ in range(self._n_mon_actions)]
            for _ in range(self._n_actions)
        ]
        self._mdp_critic.reset()

    def report(self):
        return self._q_dict, self._mdp_critic.report()
