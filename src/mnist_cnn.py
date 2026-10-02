"""Train a small CNN on MNIST (default hyperparameters) while logging loss, accuracy and plasticity metrics."""
import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.plasticity import (dormant_units_by_tau, effective_rank, get_relu_features, gradient_norm,  # noqa: E402
                            percent_active_parameters, srank, unit_sign_entropy, weight_norm)


class MnistCNN(nn.Module):
    """2 conv layers + 2 MLP layers."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 24 * 24, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device) -> tuple:
    model.eval()
    loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.numel()
    return loss / n, 100 * correct / n


def plasticity_metrics(model, probe, probe_y, taus=(0.0, 0.025, 0.1)) -> dict:
    """`dormant_pct` / `dormant_pct_layer{i}` are always tau=0.025 (kept for backward compatibility)."""
    by_tau = dormant_units_by_tau(model, probe, tuple(sorted(set(taus) | {0.025})))
    dormant, per_layer = by_tau[0.025]
    last_hidden = get_relu_features(model, probe)[-1]
    sign_entropy, sign_entropy_layers = unit_sign_entropy(model, probe)
    metrics = {
        "dormant_pct": dormant,
        "active_params_pct": percent_active_parameters(model, 1e-3),
        "effective_rank": effective_rank(last_hidden),
        "srank": srank(last_hidden),
        "unit_sign_entropy": sign_entropy,
        "grad_norm": gradient_norm(model, probe, probe_y),
        "weight_norm": weight_norm(model),
    }
    metrics.update({f"dormant_pct_tau_{tau:g}": by_tau[tau][0] for tau in taus})
    metrics.update({f"dormant_pct_layer{i}": v for i, v in enumerate(per_layer)})
    metrics.update({f"unit_sign_entropy_layer{i}": v for i, v in enumerate(sign_entropy_layers)})
    return metrics


def plot_results(df: pd.DataFrame, step_losses: list, out_dir: str):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(step_losses, lw=0.7)
    axes[0].set(xlabel="Step", ylabel="Train loss (per step)")
    axes[1].plot(df.epoch, df.train_loss, label="train")
    axes[1].plot(df.epoch, df.test_loss, label="test")
    axes[1].set(xlabel="Epoch", ylabel="Loss")
    axes[2].plot(df.epoch, df.train_acc, label="train")
    axes[2].plot(df.epoch, df.test_acc, label="test")
    axes[2].set(xlabel="Epoch", ylabel="Accuracy (%)")
    axes[3].plot(df.epoch, df.dormant_pct, label="dormant units (%)")
    axes[3].plot(df.epoch, df.active_params_pct, label="active params (%)")
    axes[3].set(xlabel="Epoch", ylabel="Percent")
    for ax in axes[1:]:
        ax.legend()
    for ax in axes:
        ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mnist_training.png"), dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="results/mnist")
    parser.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.025, 0.1], help="dormant thresholds to log")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Each run gets its own timestamped folder: <out_dir>/<YYYYmmdd_HHMMSS>/{checkpoints, metrics, plots}
    run_dir = os.path.join(args.out_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    device = get_device()

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=tf)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    train_eval_loader = torch.utils.data.DataLoader(train_set, batch_size=1000)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)
    # Fixed probe batch (from the test set) for plasticity metrics
    probe = torch.stack([test_set[i][0] for i in range(1024)]).to(device)
    probe_y = torch.tensor([test_set[i][1] for i in range(1024)]).to(device)

    model = MnistCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    rows, step_losses = [], []

    def log(epoch):
        train_loss, train_acc = evaluate(model, train_eval_loader, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
               "test_loss": test_loss, "test_acc": test_acc, **plasticity_metrics(model, probe, probe_y, tuple(args.taus))}
        rows.append(row)
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
                         if not k.startswith("dormant_pct_layer")), flush=True)
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "metrics": row},
                   os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pt"))

    log(0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            step_losses.append(loss.item())
        log(epoch)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    np.save(os.path.join(run_dir, "step_losses.npy"), np.array(step_losses))
    plot_results(df, step_losses, run_dir)
    torch.save(model.state_dict(), os.path.join(run_dir, "model_final.pt"))
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
