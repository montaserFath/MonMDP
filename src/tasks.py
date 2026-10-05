"""Shuffled-MNIST tasks (Dohare et al. 2024 style): a task is a fixed pixel permutation or class-id permutation.

Task 0 is always the identity (plain MNIST). Permutations are seeded from (run_seed, task_idx).
"""
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms

N_PIXELS = 28 * 28


def load_mnist(data_dir: str, device: torch.device) -> tuple:
    """Whole MNIST as normalised tensors on the device: (x_train, y_train, x_test, y_test)."""
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    out = []
    for train in (True, False):
        ds = datasets.MNIST(data_dir, train=train, download=True, transform=tf)
        x = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
        y = torch.tensor([ds[i][1] for i in range(len(ds))]).to(device)
        out += [x, y]
    return tuple(out)


def make_task(kind: str, run_seed: int, task_idx: int) -> dict:
    """kind in {'pixel', 'label'}. Returns {'kind', 'pixel_perm', 'label_perm'} (None where unused)."""
    assert kind in ("pixel", "label"), kind
    gen = torch.Generator().manual_seed(run_seed * 100003 + task_idx)
    n = N_PIXELS if kind == "pixel" else 10
    perm = torch.arange(n) if task_idx == 0 else torch.randperm(n, generator=gen)
    return {"kind": kind, "pixel_perm": perm if kind == "pixel" else None,
            "label_perm": perm if kind == "label" else None}


def apply_task(x: torch.Tensor, y: torch.Tensor, task: dict) -> tuple:
    """Apply the same permutation to every image (pixel task) or to every label (label task)."""
    if task["pixel_perm"] is not None:
        perm = task["pixel_perm"].to(x.device)
        x = x.flatten(1)[:, perm].reshape(x.shape)
    if task["label_perm"] is not None:
        y = task["label_perm"].to(y.device)[y]
    return x, y


@torch.no_grad()
def evaluate_tensors(model, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1000) -> tuple:
    """(loss, accuracy %) of the model on tensors."""
    was_training = model.training
    model.eval()
    loss, correct = 0.0, 0
    for i in range(0, len(x), batch_size):
        out = model(x[i:i + batch_size])
        loss += F.cross_entropy(out, y[i:i + batch_size], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + batch_size]).sum().item()
    model.train(was_training)
    return loss / len(x), 100 * correct / len(x)
