import random
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt

from src.policy_analysis import plot_env_actions
from src.wrappers.env_wrappers import ChannelsObs, TimeStepReward, WallObs, WindowViewObs
from src.wrappers.monitor_wrappers import BinaryMonitor

LOG_DIR = "../cc_general_models/9_9/Penalty/reward_model/env_0.0/eps_decay/window_7/q_lr_0.0001/reward_lr_0.001/"
GRID_SIZE = (9, 9)
DEVICE = "mps:0"
WINDOW_SIZE = 7
N_MDP_ACTIONS = 4


def select_action(q_values) -> int:
    """Choose an action from a Q value distribution/ break ties to a random action"""
    indices = np.argwhere(q_values == np.max(q_values))
    return random.choice(indices)[0]


def plot_policy(
        q_model, grid_size: (int, int), seed: int, n_mdp_actions: int, window: bool = False, save_fig: bool = False,
) -> None:
    """Plot policy actions for each state"""
    actions_ind = []
    penalty_vec = np.load("../cc_general_models/9_9/Penalty/reward_model/env_0.0/obs_vector.npy")
    actions = np.zeros((int(grid_size[0] * grid_size[1]), 2))
    q_values = np.zeros((int(grid_size[0] * grid_size[1]), 8))
    count = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            vec = torch.zeros((1, 9, 9), device=DEVICE)
            vec[0, i, j] = 1
            obs = torch.cat((vec, torch.from_numpy(penalty_vec).to(DEVICE)))
            if window:
                obs = add_walls(obs, n_walls=3)
                obs = add_window(obs, window_size=WINDOW_SIZE)
            with torch.no_grad():
                q_values[count] = q_model(obs.unsqueeze(0)).detach().cpu().numpy().squeeze()
                actions_ind.append(select_action(q_values[count]))
            actions[count, 0], actions[count, 1] = actions_ind[-1] % n_mdp_actions, actions_ind[-1] // n_mdp_actions
            count += 1
    fig = plot_env_actions("Penalty", states=np.arange(grid_size[0] * grid_size[1]), actions=actions, grid_size=(9, 9))
    if save_fig:
        fig.savefig(LOG_DIR + "/final_policy_plot_{}".format(seed), dpi=300)


def add_walls(obs: torch.tensor, n_walls: int) -> torch.tensor:
    """Add walls to the observation space"""
    new_obs = torch.zeros((obs.shape[0] + 1, obs.shape[1] + 2 * n_walls, obs.shape[2] + 2 * n_walls), device=DEVICE)
    new_obs[:-1, n_walls:-n_walls, n_walls:-n_walls] = obs.clone()
    new_obs[-1, :n_walls, :] = 1
    new_obs[-1, :, :n_walls] = 1
    new_obs[-1, -n_walls:, :] = 1
    new_obs[-1, :, -n_walls:] = 1

    return new_obs


def add_window(obs: torch.tensor, window_size: int) -> torch.tensor:
    """Change the observation space to be a window centred around the agent location"""
    agent_id = 1  # TODO remove hard coded value
    obs = obs.detach().cpu().numpy()
    pos_x, pos_y = np.where(obs[0, :, :] == agent_id)[0][0], np.where(obs[0, :, :] == agent_id)[1][0]
    min_x, max_x = max(0, pos_x - window_size // 2), min(obs.shape[1], pos_x + window_size // 2 + 1)
    min_y, max_y = max(0, pos_y - window_size // 2), min(obs.shape[2], pos_y + window_size // 2 + 1)
    window_obs = obs[:, min_x:max_x, min_y:max_y]
    return torch.from_numpy(window_obs).to(DEVICE)


def eval_policy(env, q_model, n_episodes: int, gamma: float = 0.99) -> (np.ndarray, np.ndarray):
    """Evaluate a policy given an environment and save episode joint reward & episode length"""
    ep_reward, ep_timestep = [], []
    for i in range(n_episodes):
        s, _ = env.reset()
        total_r, timestep = 0, 0
        done = False
        while not done and timestep < 5000:
            with torch.no_grad():
                q_values = q_model(torch.tensor(s["mdp"], dtype=torch.float32, device=DEVICE).unsqueeze(0))
            action = select_action(q_values.cpu().numpy().squeeze())  # q_values.max(1).indices.view(1, 1).item()
            s, r, done, _, info = env.step({"mdp": action % 4, "monitor": action // 4})
            reward = info["mdp_reward"] + r["monitor"]
            total_r += reward * (gamma**timestep)
            timestep += 1
        ep_reward.append(total_r)
        ep_timestep.append(timestep)
    return np.array(ep_reward), np.array(ep_timestep)


def evaluate_police_seeds_checkpoints(
    n_seeds: int,
    n_checkpoints: int,
    n_episodes: int,
    n_episodes_checkpoint: int,
    window_obs: bool = False,
    save_fig: bool = False,
) -> None:
    """Evaluate police seeds checkpoints and plot episode joint reward and plot policy actions"""
    x_axis = n_episodes_checkpoint * np.arange(n_checkpoints)
    for seed in range(n_seeds):
        env = gym.make("gym_monitor/TreasureHunt-Penalty-v1", render_modes="human", seed=seed)
        env = ChannelsObs(env, grid_size=GRID_SIZE, n_objects=3)
        if window_obs:
            env = WallObs(env, grid_size=GRID_SIZE, n_walls=3)
            env = WindowViewObs(env, window_size=WINDOW_SIZE)
        env = TimeStepReward(env, timestep_penalty=0.0, goal_reward=1, fire_reward=-1)
        env = BinaryMonitor(env, full_monitor=False, monitor_cost=0.2, init_monitor_state=0, monitor_reset_prob=0.0)

        eval_rewards, eval_timestep = [], []
        for checkpoint in range(n_checkpoints):
            q_model = torch.load(
                LOG_DIR + "/checkpoints_{}/q_network_{}".format(checkpoint, seed), map_location=torch.device(DEVICE)
            )
            ep_r, ep_t = eval_policy(env, q_model, n_episodes)
            eval_rewards.append(ep_r)
            eval_timestep.append(ep_t)

        np.save(LOG_DIR + "/eval_joint_reward_{}".format(seed), np.squeeze(eval_rewards))
        np.save(LOG_DIR + "/eval_episode_length_{}".format(seed), np.squeeze(eval_timestep))
        plot_policy(q_model, grid_size=GRID_SIZE, n_mdp_actions=N_MDP_ACTIONS, window=True, seed=seed, save_fig=save_fig)
        fig = plt.figure(figsize=(6, 4))
        plt.plot(x_axis, np.squeeze(eval_rewards), lw=2)
        plt.scatter(x_axis, np.squeeze(eval_rewards), marker="o", s=10)
        plt.hlines(0.79, 0, x_axis[-1], color="r", lw=3, linestyle="--")  # optimal episode joint reward
        plt.xlabel("Training Timesteps")
        plt.ylabel("Eval Discounted Joint Episode Reward")
        plt.grid(axis="y")
        plt.tight_layout()
        if save_fig:
            fig.savefig(LOG_DIR + "/checkpoint_eval_joint_reward_{}.png".format(seed), dpi=300)


if __name__ == "__main__":
    evaluate_police_seeds_checkpoints(
        n_seeds=10,
        n_checkpoints=43,
        n_episodes=1,
        window_obs=True,
        n_episodes_checkpoint=int(5e4),
        save_fig=True,
    )
