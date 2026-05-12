"""helper functions for k-of-N analysis"""
import numpy as np
import torch
from src.replay_buffer import TorchReplayMemory


# pylint: disable=too-many-locals
def load_replay_buffer(grid_size: (int, int), window_size: int, dryness: float, seed: int, device: str) -> dict:
    """Load replay buffer from disk & separate it by plant type"""
    q_lr = 0.0001
    r_lr = 0.0001

    main_dir = (
        f"../general_models/Plants/reward_model/{grid_size[0]}_{grid_size[1]}_channel/dry_{dryness}/3_room/"
        + f"window_{window_size}/eps_decay_1e-07/q_lr_{q_lr}/reward_lr_{r_lr}"
    )
    replay_buffer = TorchReplayMemory(int(1e6))
    replay_buffer.load(main_dir + "/checkpoints_3/", seed)

    batch = replay_buffer.process_batch(replay_buffer.sample(len(replay_buffer) - 1), device=device)
    unmonitored_ind = np.where(batch["mon_obs"].cpu().numpy() == 1)[0]
    plant_channels = batch["mdp_obs"][unmonitored_ind][:, 1:4, window_size // 2, window_size // 2].cpu().numpy()

    plant_index = []
    cactus_index = []
    diff_1_index, diff_2_index, diff_3_index, diff_4_index, diff_5_index = [], [], [], [], []
    for i, row in enumerate(plant_channels):
        if (row == np.array([0, 1, 1]) / 2).all():
            cactus_index.append(i)
        elif (row == np.array([1, 1, 0]) / 2).all():
            plant_index.append(i)
        elif (row == np.array([0, 0, 1])).all():
            diff_1_index.append(i)
        elif (row == np.array([0, 1, 0])).all():
            diff_2_index.append(i)
        elif (row == np.array([1, 0, 0])).all():
            diff_3_index.append(i)
        elif (row == np.array([1, 0, 1]) / 2).all():
            diff_4_index.append(i)
        elif np.allclose(row, np.array([1, 1, 1]) / 3):
            diff_5_index.append(i)
        else:
            continue

    return {
        "plant": batch["mdp_obs"][unmonitored_ind][plant_index],
        "cactus": batch["mdp_obs"][unmonitored_ind][cactus_index],
        "diff_1": batch["mdp_obs"][unmonitored_ind][diff_1_index],
        "diff_2": batch["mdp_obs"][unmonitored_ind][diff_2_index],
        "diff_3": batch["mdp_obs"][unmonitored_ind][diff_3_index],
        "diff_4": batch["mdp_obs"][unmonitored_ind][diff_4_index],
        "diff_5": batch["mdp_obs"][unmonitored_ind][diff_5_index],
    }


def get_obs_by_dryness_level(window_size: int, obs: dict, dry_value: float) -> dict:
    """separate observations by dryness level"""
    obs_dryness = {}
    midd_ind = window_size // 2
    j = np.where(obs["plant"][:, 4, midd_ind, midd_ind].detach().cpu().numpy() == dry_value)[0]
    obs_dryness.update({"plant": obs["plant"][j]})

    j = np.where(obs["cactus"][:, 4, midd_ind, midd_ind].detach().cpu().numpy() == dry_value)[0]
    obs_dryness.update({"cactus": obs["cactus"][j]})

    j = np.where(obs["diff_1"][:, 4, midd_ind, midd_ind].detach().cpu().numpy() == dry_value)[0]
    obs_dryness.update({"diff_1": obs["diff_1"][j]})

    j = np.where(obs["diff_2"][:, 4, midd_ind, midd_ind].detach().cpu().numpy() == dry_value)[0]
    obs_dryness.update({"diff_2": obs["diff_2"][j]})

    j = np.where(obs["diff_3"][:, 4, midd_ind, midd_ind].detach().cpu().numpy() == dry_value)[0]
    obs_dryness.update({"diff_3": obs["diff_3"][j]})

    j = np.where(obs["diff_4"][:, 4, midd_ind, midd_ind].detach().cpu().numpy() == dry_value)[0]
    obs_dryness.update({"diff_4": obs["diff_4"][j]})

    j = np.where(obs["diff_5"][:, 4, midd_ind, midd_ind].detach().cpu().numpy() == dry_value)[0]
    obs_dryness.update({"diff_5": obs["diff_5"][j]})
    return obs_dryness


def load_reward_models(grid_size: (int, int), window_size: int, dryness: float, n_r_models: int) -> list:
    """Load reward models"""
    q_lr = 0.0001
    r_lr = 0.0001
    main_dir = (
        f"../general_models/Plants/reward_model/{grid_size[0]}_{grid_size[1]}_channel/dry_{dryness}/room/"
        + f"window_{window_size}/q_lr_{q_lr}/reward_lr_{r_lr}"
    )
    trained_models = []
    for r in range(n_r_models):
        trained_models.append(torch.load(main_dir + f"/ensemble_reward_models/reward_model_{r}"))
    return trained_models


# pylint: disable=protected-access
def get_rewards(window_size: int, n_mdp_actions: int, trained_models: list, obs: dict) -> dict:
    """Get reward model predictions given observations"""
    midd_ind = window_size // 2
    n_r_models = len(trained_models)

    plants_reward = np.zeros((len(obs["plant"]), n_r_models, n_mdp_actions))
    cactus_reward = np.zeros((len(obs["cactus"]), n_r_models, n_mdp_actions))
    diff_1_reward = np.zeros((len(obs["diff_1"]), n_r_models, n_mdp_actions))
    diff_2_reward = np.zeros((len(obs["diff_2"]), n_r_models, n_mdp_actions))
    diff_3_reward = np.zeros((len(obs["diff_3"]), n_r_models, n_mdp_actions))
    diff_4_reward = np.zeros((len(obs["diff_4"]), n_r_models, n_mdp_actions))
    diff_5_reward = np.zeros((len(obs["diff_5"]), n_r_models, n_mdp_actions))
    for r, r_model in enumerate(trained_models):
        plants_reward[:, r] = r_model._network(obs["plant"][:, :, midd_ind, midd_ind]).detach().cpu().numpy()
        cactus_reward[:, r] = r_model._network(obs["cactus"][:, :, midd_ind, midd_ind]).detach().cpu().numpy()
        diff_1_reward[:, r] = r_model._network(obs["diff_1"][:, :, midd_ind, midd_ind]).detach().cpu().numpy()
        diff_2_reward[:, r] = r_model._network(obs["diff_2"][:, :, midd_ind, midd_ind]).detach().cpu().numpy()
        diff_3_reward[:, r] = r_model._network(obs["diff_3"][:, :, midd_ind, midd_ind]).detach().cpu().numpy()
        diff_4_reward[:, r] = r_model._network(obs["diff_4"][:, :, midd_ind, midd_ind]).detach().cpu().numpy()
        diff_5_reward[:, r] = r_model._network(obs["diff_5"][:, :, midd_ind, midd_ind]).detach().cpu().numpy()

    return {
        "plant": plants_reward,
        "cactus": cactus_reward,
        "diff_1": diff_1_reward,
        "diff_2": diff_2_reward,
        "diff_3": diff_3_reward,
        "diff_4": diff_4_reward,
        "diff_5": diff_5_reward,
    }
