"""Plasticity-loss metrics: dormant units (ReDo, Sokar et al. 2023), active parameters, effective rank."""
import torch
from torch import nn


def get_relu_features(model: nn.Module, x: torch.Tensor) -> list:
    """Run the model on x and return the post-ReLU output of every ReLU call, in execution order."""
    features, hooks = [], []

    def hook(_, __, output):
        features.append(output.detach().clone())

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(x)
    finally:
        model.train(was_training)
        for h in hooks:
            h.remove()
    return features


def dormant_units_proportion(model: nn.Module, x: torch.Tensor, tau: float = 0.025) -> tuple:
    """Percentage of dormant units over all hidden (ReLU) layers.

    A unit is dormant if its score s = E|h| / mean_layer(E|h|) is <= tau. Conv units are channels.
    Returns (overall_percent, per_layer_percent_list).
    """
    per_layer, n_dormant, n_units = [], 0, 0
    for feats in get_relu_features(model, x):
        dims = (0,) + tuple(range(2, feats.dim()))
        mean_abs = feats.abs().mean(dim=dims)
        score = mean_abs / (mean_abs.mean() + 1e-9)
        dormant = (score <= tau).sum().item()
        per_layer.append(100 * dormant / mean_abs.numel())
        n_dormant += dormant
        n_units += mean_abs.numel()
    return 100 * n_dormant / max(n_units, 1), per_layer


def percent_active_parameters(model: nn.Module, threshold: float = 1e-3) -> float:
    """Percentage of trainable parameters with |w| > threshold."""
    active = total = 0
    for p in model.parameters():
        if p.requires_grad:
            active += (p.detach().abs() > threshold).sum().item()
            total += p.numel()
    return 100 * active / total


def effective_rank(features: torch.Tensor, eps: float = 1e-12) -> float:
    """Entropy-based effective rank (Roy & Vetterli 2007) of a (N, D) feature matrix."""
    features = features.flatten(1).float().cpu()  # svd is not implemented on MPS
    s = torch.linalg.svdvals(features)
    p = s / (s.sum() + eps)
    return torch.exp(-(p * torch.log(p + eps)).sum()).item()


def weight_norm(model: nn.Module) -> float:
    """L2 norm of all trainable parameters."""
    return torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters() if p.requires_grad)).item()
