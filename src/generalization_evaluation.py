import random
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt

from src.policy_analysis import plot_env_actions
from src.wrappers.env_wrappers import ChannelsObs, TimeStepReward
from src.wrappers.monitor_wrappers import BinaryMonitor

LOG_DIR = "../cc_general_models/9_9/Penalty/reward_model/env_0.0/eps_decay/q_lr_0.001/reward_lr_0.001/"
GRID_SIZE = (9, 9)
DEVICE = "mps:0"


def select_action(q_values) -> int:
    """Choose an action from a Q value distribution"""
    indx = np.argwhere(q_values == np.max(q_values))
    return random.choice(indx)[0]


def plot_policy(q_model, grid_size: (int, int), seed: int, save_fig: bool = False) -> None:
    """Plot policy actions for each state"""
    actions_ind = []
    penalty_vec = np.load("../cc_general_models/9_9/Penalty/reward_model/env_0.0/obs_vector.npy")
    actions = np.zeros((int(grid_size[0] * grid_size[1]), 2))
    q_values = np.zeros((int(grid_size[0] * grid_size[1]), 8))
    count = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            vec = torch.zeros((1, 9, 9), device="mps:0")
            vec[0, i, j] = 1
            obs = torch.cat((vec, torch.from_numpy(penalty_vec).to(DEVICE)))
            with torch.no_grad():
                q_values[count] = q_model(obs.unsqueeze(0)).detach().cpu().numpy().squeeze()
                actions_ind.append(select_action(q_values[count]))
            actions[count, 0], actions[count, 1] = actions_ind[-1] % 4, actions_ind[-1] // 4
            count += 1
    fig = plot_env_actions("Penalty", states=np.arange(grid_size[0] * grid_size[1]), actions=actions, grid_size=(9, 9))
    if save_fig:
        fig.savefig(LOG_DIR + "/final_policy_plot_{}".format(seed), dpi=300)


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
    n_seeds: int, n_checkpoints: int, n_episodes: int, save_fig: bool = False
) -> None:
    """Evaluate police seeds checkpoints and plot episode joint reward and plot policy actions"""
    for seed in range(n_seeds):
        env = gym.make("gym_monitor/TreasureHunt-Penalty-v1", render_modes="human", seed=seed)
        env = ChannelsObs(env, grid_size=GRID_SIZE, n_objects=3)
        env = TimeStepReward(env, timestep_penalty=0.0, goal_reward=1, fire_reward=-10)
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
        plot_policy(q_model, grid_size=GRID_SIZE, seed=seed, save_fig=save_fig)
        fig = plt.figure(figsize=(6, 4))
        plt.plot(np.squeeze(eval_rewards), lw=2)
        plt.hlines(0.79, 0, n_checkpoints, color="r", lw=3, linestyle="--")  # optimal episode joint reward
        plt.xlabel("checkpoint")
        plt.ylabel("Eval Joint Episode Reward")
        plt.grid(axis="y")
        plt.tight_layout()
        if save_fig:
            fig.savefig(LOG_DIR + "/checkpoint_eval_joint_reward_{}.png".format(seed), dpi=300)


if __name__ == "__main__":
    evaluate_police_seeds_checkpoints(n_seeds=2, n_checkpoints=10, n_episodes=1, save_fig=True)
