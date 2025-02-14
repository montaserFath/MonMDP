"""Run k-of-N optimization"""
import numpy as np
import torch
import gymnasium as gym
from src.replay_buffer import TorchReplayMemory
from src.critic import MonRoomCNN
from src.wrappers.env_wrappers import WallObs, WindowViewObs, MultiChannel
from src.wrappers.monitor_wrappers import RoomMonitor

LOG_DIR = "../general_models/Plants/reward_model/10_10_channel/dry_0.05/3_room/window_11/eps_decay_1e-07/q_lr_0.0001/reward_lr_0.0001"
N_Models = 500
BATCH_SIZE = 128


def load_replay_buffer(buffer_dir: str) -> list:
    replay_buffer_1 = TorchReplayMemory(int(1e6))
    replay_buffer_1.load(buffer_dir + "/checkpoints_{}/".format(1), 4)

    replay_buffer_2 = TorchReplayMemory(int(1e6))
    replay_buffer_2.load(buffer_dir + "/checkpoints_{}/".format(2), 4)

    replay_buffer_3 = TorchReplayMemory(int(1e6))
    replay_buffer_3.load(buffer_dir + "/checkpoints_{}/".format(3), 4)
    return [replay_buffer_1, replay_buffer_2, replay_buffer_3]


def load_reward_models(model_dir: str, n_models: int, device: str) -> list:
    trained_models = []
    for r in range(n_models):
        trained_models.append(torch.load(
            model_dir + "/ensemble_reward_models/reward_model_{}".format(r), map_location=torch.device(device)))
    return trained_models


def prepare_critic(critic_dir: str, device: str):
    env_id = "gym_monitor/Plants-Watering-v1"
    grid_size = [10, 10]
    window_size = 11
    dryness = 0.05

    reward_prams = {
        "r0": 0,
        "lr": 1e-4,  # 1.0
        "kernel_size_0": 5,
        "kernel_size_1": 3,
        "flatten": True,
        "stride_0": 1,
        "stride_1": 1,
        "device": device,
    }

    plants_env = gym.make(
        env_id,
        grid_size=grid_size,
        n_plants=8,
        plants_dryness_prob=dryness,
        dry_difference=0.5,
        agent_start_pos=None,
        max_episode_steps=100,
        render_mode="human",
        add_new_plants=False,
        add_more_plants=True,
    )

    wall_env = WallObs(plants_env, grid_size=grid_size, n_walls=10)
    window_env = WindowViewObs(wall_env, window_size=window_size)
    channel_env = MultiChannel(window_env, normalize_obs=True)
    env = RoomMonitor(channel_env, full_monitor=False, monitor_cost=0.0, monitor_column_ind=5)

    critic = MonRoomCNN(env_id, env.observation_space, env.action_space, dir_name=critic_dir, on_policy=False,
                        q0=0,
                        gamma=0.99,
                        lr=1e-4,
                        strategy="reward_model",
                        unseen_r_value=0.0,
                        kernel_size_0=5,
                        kernel_size_1=3,
                        stride_0=1,
                        stride_1=1,
                        device=device,
                        reward_model=reward_prams,
                        )
    critic.reset()
    return critic


def sort_and_k_least(array, k):
    """sort and selects k least"""
    # return array.argsort()[:k]
    return torch.argsort(array)[:k]


def evaluate_obs(obs: torch.tensor, models: list, device: str, n_mdp_actions: int = 6) -> torch.tensor:
    """Run inference for each model given a state"""
    rewards = torch.zeros((len(models), obs.shape[0], n_mdp_actions), device=device)
    for i, model in enumerate(models):
        rewards[i] = model._network(obs)
    return rewards


def optimize_k_of_n(
        critic, buffers: list, reward_models: list, k: int, n: int, n_iterations: int, batch_size: int, device: str, seed: int,
):
    center_indx = 5
    update_target_freq = 5
    n_models = len(reward_models)
    critic.reset()
    for itr in range(n_iterations):
        replay_buffer = np.random.choice(buffers)
        sampled_batch = replay_buffer.process_batch(replay_buffer.sample(batch_size), device=device)

        indx = np.random.choice(np.arange(n_models), n, replace=False)
        selected_models = [reward_models[i] for i in indx]

        n_rewards = evaluate_obs(sampled_batch["mdp_obs"][:, :, center_indx, center_indx], selected_models, device)
        with torch.no_grad():
            qs = critic._q_network.forward(sampled_batch["mdp_obs"], sampled_batch["mon_obs"])
        policy = torch.softmax(qs, 1)
        policy_values = torch.sum(torch.sum(n_rewards * policy, 2), 1)
        k_rewards = torch.mean(n_rewards[sort_and_k_least(policy_values, k)], 0)
        _, _ = critic.optimize_policy_model(
            sampled_batch, update_target=itr % update_target_freq, monitor_state=True, rewards=k_rewards,
        )
        if itr % 1e4 == 0:
            critic.save(seed)
    critic.save(seed)


def train_k_of_n(k: int, n: int, n_iterations: int, batch_size: int, device: str, seed: int) -> None:
    critic_str = LOG_DIR + f"/{k}_of_{n}/"
    buffers = load_replay_buffer(LOG_DIR)
    reward_models = load_reward_models(LOG_DIR, N_Models, device)
    critic = prepare_critic(critic_str, device)
    optimize_k_of_n(critic, buffers, reward_models, k, n, n_iterations, batch_size, device, seed)


if __name__ == "__main__":
    for k in [1, 5, 10]:
        for seed in range(10):
            train_k_of_n(k, 10, int(5e4), BATCH_SIZE, device="mps:0", seed=seed + 20)
