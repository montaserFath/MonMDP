"""Plotting functions for the training/testing, heatmaps for the q-tables, policy actions, and final trajectories"""
import numpy as np
import colorsys
import seaborn as sns


def set_blind_colors() -> (list, list):
    """Get Blind colors friendly colors"""
    dark_hues = [0, 0.1, 0.4, 0.55, 0.65, 0.75, 0.9]
    dark_lightness = [0.4, 0.2, 0.15, 0.4, 0.4, 0.4, 0.4]
    light_hues = [1, 0.1, 0.3, 0.5, 0.6, 0.75, 0.85]
    light_lightness = [0.75, 0.6, 0.45, 0.6, 0.65, 0.8, 0.55]
    dark_colors = [colorsys.hls_to_rgb(h, dark_lightness[i], 1) for i, h in enumerate(dark_hues)]
    sns.palplot(dark_colors)
    light_colors = [colorsys.hls_to_rgb(h, light_lightness[i], 1) for i, h in enumerate(light_hues)]
    sns.palplot(light_colors)
    return light_colors, dark_colors


def discount_episode_reward(reward: dict, gamma: float = 0.99, length: int = 5000) -> (np.ndarray, np.ndarray):
    """Calculate each episode discount reward and number of timesteps"""
    discount = [gamma**i for i in range(length)]
    discount_reward, length = np.zeros(len(reward)), np.zeros(len(reward))
    for i, key in enumerate(reward.keys()):
        length[i] = len(reward[key])
        discount_reward[i] = np.sum(np.array(reward[key]) * np.array(discount[: len(reward[key])]))
    return discount_reward, length


def sum_ep_timesteps(ep_timesteps: np.ndarray) -> np.ndarray:
    """add number of timesteps for each episode"""
    sum_timesteps = np.zeros(len(ep_timesteps))
    last_length = 0
    for i, timestep in enumerate(ep_timesteps):
        sum_timesteps[i] = last_length + timestep
        last_length = sum_timesteps[i]
    return sum_timesteps


def ind_to_action(action_ind: int, n_mdp_actions: int = 4) -> (int, int):
    """Transform integer index to MDP and Monitor actions"""
    mon_action, mdp_action = action_ind // n_mdp_actions, action_ind % n_mdp_actions
    return mdp_action, mon_action


def get_action_ind(action: dict, n_mdp_actions: int = 4) -> int:
    """convert a dictionary of MDP and Monitor actions to an index"""
    mdp_action, mon_action = action["mdp"], action["monitor"]
    return mon_action * n_mdp_actions + mdp_action
