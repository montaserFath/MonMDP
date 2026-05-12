"""Network analysis utilities: dormant neuron detection and overgeneralization measurement."""
from typing import List, Tuple

import numpy as np
import torch


def generate_half_room_states(
    n_samples: int,
    n_plants: int = 4,
    n_cacti: int = 4,
    dryness_levels: Tuple[float, ...] = (0.0, 0.5, 1.0),
    grid_size: int = 11,
    normalize_obs: bool = True,
    plant_cactus: bool = False,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates random full-window observations (6, 11, 11) for the
    monitored and unmonitored halves of the half-room environment.

    Observation channels: [agent, plant_0, plant_1, plant_2, dryness, wall]
    Plant encoding: (1, 1, 0), cactus encoding: (0, 1, 1).

    The underlying grid is 10x10, split into:
      - monitored zone   (cols 0-4): plants only
      - unmonitored zone (cols 5-9): plants and cacti

    obs_states:   agent in monitored zone — 4 plants on the left side
    unobs_states: agent in unmonitored zone — 4 plants + 4 cacti on the
                  right side (when plant_cactus=True)

    Args:
        n_samples: number of observations per batch
        n_plants: number of regular plants (default 4)
        n_cacti: number of cacti in unmonitored zone (default 4, used when plant_cactus=True)
        dryness_levels: possible dryness values (default (0.0, 0.5, 1.0))
        grid_size: spatial size of the window (default 11)
        normalize_obs: if True, plant channels are halved (0.5, 0.5, 0) (default True)
        plant_cactus: if True, unmonitored zone contains cacti alongside plants
        device: torch device string

    Returns:
        (obs_states, unobs_states) each of shape (n_samples, 6, grid_size, grid_size)
    """
    center = grid_size // 2
    div = 2.0 if normalize_obs else 1.0
    plant_channels = torch.tensor([1.0 / div, 1.0 / div, 0.0])
    cactus_channels = torch.tensor([0.0, 1.0 / div, 1.0 / div])
    dryness_opts = torch.tensor(dryness_levels)

    left_cells = [
        (r, c) for r in range(grid_size) for c in range(center)
    ]
    right_cells = [
        (r, c) for r in range(grid_size) for c in range(center + 1, grid_size)
    ]

    def _place(batch, i, r, c, channels):
        batch[i, 1:4, r, c] = channels
        batch[i, 4, r, c] = dryness_opts[torch.randint(len(dryness_opts), (1,))]

    def _build_obs(n: int) -> torch.Tensor:
        """Monitored zone: n_plants plants on left side only."""
        batch = torch.zeros(n, 6, grid_size, grid_size)
        batch[:, 0, center, center] = 1.0

        for i in range(n):
            idx = torch.randperm(len(left_cells))[:n_plants]
            for j in idx:
                r, c = left_cells[j]
                _place(batch, i, r, c, plant_channels)

            if torch.rand(1).item() > 0.5:
                _place(batch, i, center, center, plant_channels)

        return batch.to(device)

    def _build_unobs(n: int) -> torch.Tensor:
        """Unmonitored zone: n_plants plants + n_cacti cacti on right side."""
        n_veg = n_plants + (n_cacti if plant_cactus else 0)
        batch = torch.zeros(n, 6, grid_size, grid_size)
        batch[:, 0, center, center] = 1.0

        for i in range(n):
            idx = torch.randperm(len(right_cells))[:n_veg]
            for j, pos_idx in enumerate(idx):
                r, c = right_cells[pos_idx]
                if plant_cactus and j >= n_plants:
                    _place(batch, i, r, c, cactus_channels)
                else:
                    _place(batch, i, r, c, plant_channels)

            roll = torch.rand(1).item()
            if roll > 0.5:
                if plant_cactus and torch.rand(1).item() > 0.5:
                    _place(batch, i, center, center, cactus_channels)
                else:
                    _place(batch, i, center, center, plant_channels)

        return batch.to(device)

    obs_states = _build_obs(n_samples)
    unobs_states = _build_unobs(n_samples)
    return obs_states, unobs_states


def get_features(
    model: torch.nn.Module,
    obs_env: torch.Tensor,
    obs_mon: torch.Tensor = None,
) -> List[torch.Tensor]:
    """
    Extract post-activation features from each layer of a PyTorch model.

    Args:
        model: nn.Sequential
        obs_env: input tensor — single (C, H, W) or batch (N, C, H, W)
        obs_mon: optional monitor observation — scalar or (N,)
    Returns:
        List of activation tensors, one per ReLU layer
    """
    if obs_env.dim() == 3:
        obs_env = obs_env.unsqueeze(0)

    features = []
    hooks = []

    def register_hook(module):
        def hook(_, __, output):
            features.append(output.detach())
        return hook

    for _, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            hooks.append(module.register_forward_hook(register_hook(module)))

    with torch.no_grad():
        if obs_mon is not None:
            if obs_mon.dim() == 0:
                obs_mon = obs_mon.unsqueeze(0)
            model[5:](torch.concat((model[:5](obs_env), obs_mon.view(-1, 1)), 1))
        else:
            model(obs_env)

    for h in hooks:
        h.remove()

    return features


def compute_dormant_units_proportion(
    model: torch.nn.Module,
    obs_env: torch.Tensor,
    dormant_unit_threshold: float = 0.01,
    obs_mon: torch.Tensor = None,
) -> float:
    """
    Computes the proportion of dormant units across all hidden layers.

    A neuron is dormant when its normalized mean absolute activation
    falls below the threshold (Sokar et al., 2023).

    Args:
        model: nn.Sequential network
        obs_env: input tensor — single (C, H, W) or batch (N, C, H, W)
        dormant_unit_threshold: activation score below which a neuron is dormant
        obs_mon: optional monitor observation — scalar or (N,)
    Returns:
        Percentage of dormant units (0-100)
    """
    features_per_layer = get_features(model, obs_env, obs_mon)

    total_dormant = 0
    total_units = 0

    for layer_idx in range(len(features_per_layer) - 1):
        feats = features_per_layer[layer_idx]

        if len(feats.shape) > 2:
            mean_abs = feats.abs().mean(dim=(0, 2, 3))
        else:
            mean_abs = feats.abs().mean(dim=0)

        normalized = mean_abs / (mean_abs.mean() + 1e-9)

        total_dormant += (normalized < dormant_unit_threshold).sum().item()
        total_units += mean_abs.shape[0]

    return 100 * total_dormant / total_units


def _true_reward_from_obs(center_pixel: torch.Tensor) -> torch.Tensor:
    """
    Derives the true reward for all 6 actions from the center pixel of the
    MultiChannel observation. The 6 channels at center are:
      [0] agent, [1] plant_0, [2] plant_1, [3] plant_2, [4] dryness, [5] wall
    Plant encoding: (1, 1, 0) for plant, (0, 0, 0) for empty.

    Reward rules (plant watering env):
      actions 0-3 (move), 5 (nothing) → 0.0
      action 4 (water):
        cactus (plant_2 > 0)          → -1.0
        dry plant (dryness > 0)       → +1.0
        watered plant (dryness == 0)  → -1.0
        no plant (all zero)           → -0.2

    Args:
        center_pixel: (N, 6) center pixel features
    Returns:
        (N, 6) true rewards for each action
    """
    N = center_pixel.shape[0]
    rewards = torch.zeros(N, 6, device=center_pixel.device)

    has_plant = center_pixel[:, 1:4].abs().sum(dim=1) > 1e-6
    is_cactus = center_pixel[:, 3].abs() > 1e-6
    is_dry = center_pixel[:, 4] > 1e-6

    water_reward = torch.where(
        is_cactus,
        torch.tensor(-1.0, device=center_pixel.device),
        torch.where(
            has_plant & is_dry,
            torch.tensor(1.0, device=center_pixel.device),
            torch.where(
                has_plant & ~is_dry,
                torch.tensor(-1.0, device=center_pixel.device),
                torch.tensor(-0.2, device=center_pixel.device),
            ),
        ),
    )
    rewards[:, 4] = water_reward
    return rewards


def compute_overgeneralization(
    reward_model: torch.nn.Module,
    obs_states: torch.Tensor,
    unobs_states: torch.Tensor,
    flatten: bool = True,
) -> np.ndarray:
    """
    Measures overgeneralization as the ratio of reward model MSE on
    unobservable states to MSE on observable states, using the
    half-room reward function to derive true rewards.

    Evaluates all 6 actions: [0, 1, 2, 3, 4, 5].

    Unobservable MSE: E[(R_hat(s,a) - r_true)^2 | unobservable]
    Observable MSE:   E[(R_hat(s,a) - r_true)^2 | observable]

    Args:
        reward_model: the reward prediction network
        obs_states: (N_obs, C, H, W) observable environment observations
        unobs_states: (N_unobs, C, H, W) unobservable environment observations
        flatten: if True, extract center pixel before feeding to model

    Returns:
        np.ndarray of shape (3,): [mse_unobservable, mse_observable, ratio]
    """
    with torch.no_grad():
        results = {}
        for label, states in [("obs", obs_states), ("unobs", unobs_states)]:
            if states.numel() == 0:
                results[label] = float("nan")
                continue

            if states.dim() == 3:
                states = states.unsqueeze(0)

            size = states.shape[-1] // 2
            center = states[:, :, size, size]

            model_input = center if flatten else states
            predictions = reward_model(model_input)
            true_rewards = _true_reward_from_obs(center)

            sq_errors = (predictions - true_rewards) ** 2
            results[label] = sq_errors.mean().item()

        mse_obs = results["obs"]
        mse_unobs = results["unobs"]

        if np.isnan(mse_obs) or mse_obs == 0:
            ratio = float("inf")
        else:
            ratio = mse_unobs / mse_obs

    return np.array([mse_unobs, mse_obs, ratio])
