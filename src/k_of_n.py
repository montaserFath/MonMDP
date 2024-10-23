"""Run k-of-N optimization"""
import numpy as np


def sort_and_k_least(array, k):
    """sort and selects k least"""
    return array.argsort()[:k]


def evalaute_obs(obs: np.ndarray, models: list, n_mdp_actions: int) -> np.ndarray:
    """Run inference for each model given a state"""
    rewards = np.zeros((len(models), n_mdp_actions))
    for i, model in enumerate(models):
        rewards[i] = model._network(obs).detach().cpu().numpy()
    return rewards


def k_of_n_policy(obs: np.ndarray, k: int, n: int, n_iterations: int, n_mdp_actions: int, models: list):
    """Get k-of-N optimal policy given a state"""
    obs_reward_value = evalaute_obs(obs, models, n_mdp_actions)
    action = (1 / n_mdp_actions) * np.ones((1, n_mdp_actions))
    total_regret = np.zeros((1, n_mdp_actions))
    expected_kof_n = []
    for _ in range(n_iterations):
        n_models = obs_reward_value[np.random.choice(len(models), n)]
        k_index = sort_and_k_least(np.sum(n_models * action, 1), k)
        mean_k = np.mean(n_models[k_index], 0)
        value_mean_k = np.sum(mean_k * action)

        expected_kof_n.append(value_mean_k)
        regret = mean_k - value_mean_k
        total_regret += regret
        action = np.maximum(0, total_regret) / np.sum(np.maximum(0, total_regret))
    return expected_kof_n, action[0]
