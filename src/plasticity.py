"""Plasticity-loss metrics: dormant units (ReDo, Sokar et al. 2023), stable rank (Kumar et al. 2021),
unit sign entropy (Lyle et al. 2024), gradient/weight norms, active parameters, effective rank."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def _collect_relu(model: nn.Module, x: torch.Tensor, pre: bool) -> list:
    """Run the model on x and return the input (pre=True) or output (pre=False) of every ReLU call."""
    captured, hooks = [], []

    def hook(_, inputs, output=None):
        captured.append((inputs[0] if pre else output).detach().clone())

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_pre_hook(hook) if pre else module.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(x)
    finally:
        model.train(was_training)
        for h in hooks:
            h.remove()
    return captured


def get_relu_features(model: nn.Module, x: torch.Tensor) -> list:
    """Post-ReLU output of every ReLU call, in execution order."""
    return _collect_relu(model, x, pre=False)


def get_relu_preactivations(model: nn.Module, x: torch.Tensor) -> list:
    """Pre-ReLU input of every ReLU call, in execution order."""
    return _collect_relu(model, x, pre=True)


def dormant_scores(model: nn.Module, x: torch.Tensor) -> list:
    """Per-layer ReDo scores s = E|h| / mean_layer(E|h|), one (n_units,) tensor per hidden layer."""
    scores = []
    for feats in get_relu_features(model, x):
        dims = (0,) + tuple(range(2, feats.dim()))
        mean_abs = feats.abs().mean(dim=dims)
        scores.append(mean_abs / (mean_abs.mean() + 1e-9))
    return scores


def dormant_units_by_tau(model: nn.Module, x: torch.Tensor, taus=(0.0, 0.025, 0.1)) -> dict:
    """Dormant-unit percentages for several thresholds from a single forward pass.

    A unit is tau-dormant if its score is <= tau. tau=0 is the dead-unit case and is tested as
    "score close to zero" (<= 1e-8), as in the ReDo reference code. Conv units are channels.
    Returns {tau: (overall_percent, per_layer_percent_list)}.
    """
    scores = dormant_scores(model, x)
    out = {}
    for tau in taus:
        thr = tau if tau > 0 else 1e-8
        counts = [(s <= thr).sum().item() for s in scores]
        sizes = [s.numel() for s in scores]
        out[tau] = (100 * sum(counts) / max(sum(sizes), 1), [100 * c / n for c, n in zip(counts, sizes)])
    return out


def dormant_units_proportion(model: nn.Module, x: torch.Tensor, tau: float = 0.025) -> tuple:
    """Percentage of tau-dormant units over all hidden (ReLU) layers: (overall_percent, per_layer_percent_list)."""
    return dormant_units_by_tau(model, x, (tau,))[tau]


def srank(features: torch.Tensor, delta: float = 0.01) -> int:
    """Stable rank of Kumar et al. (2021): smallest k whose top-k singular values hold >= (1 - delta)
    of the total singular-value sum of the (N, D) feature matrix."""
    s = torch.linalg.svdvals(features.flatten(1).float().cpu())  # svd is not implemented on MPS
    cum = torch.cumsum(s, 0) / s.sum().clamp_min(1e-12)
    return int((cum < 1 - delta).sum().item()) + 1


def gradient_norm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn=F.cross_entropy) -> float:
    """Global L2 norm of the loss gradient on batch (x, y). Uses autograd.grad, so .grad buffers are untouched."""
    params = [p for p in model.parameters() if p.requires_grad]
    was_training = model.training
    model.eval()
    try:
        grads = torch.autograd.grad(loss_fn(model(x), y), params)
    finally:
        model.train(was_training)
    return torch.sqrt(sum((g ** 2).sum() for g in grads)).item()


def unit_sign_entropy(model: nn.Module, x: torch.Tensor) -> tuple:
    """Average unit sign entropy (bits, max 1) over hidden units, from the sign of pre-ReLU activations.

    Per unit, p = P(pre-activation > 0) over inputs (conv: over inputs and spatial positions) and the
    entropy is H(p). Low values mean units that are saturated (always off) or linearized (always on).
    Returns (mean_over_all_units, per_layer_means).
    """
    per_layer, total, n_units = [], 0.0, 0
    for pre in get_relu_preactivations(model, x):
        if pre.dim() > 2:  # (N, C, ...) -> (N * positions, C)
            pre = pre.transpose(1, -1).reshape(-1, pre.shape[1])
        p = (pre > 0).float().mean(0)
        ent = -(torch.special.xlogy(p, p) + torch.special.xlogy(1 - p, 1 - p)) / math.log(2)  # 0*log(0) = 0
        per_layer.append(ent.mean().item())
        total += ent.sum().item()
        n_units += ent.numel()
    return total / max(n_units, 1), per_layer


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
