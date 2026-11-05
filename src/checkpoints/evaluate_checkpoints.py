"""Recompute plasticity metrics for every saved checkpoint of one or more MNIST runs.

Usage: python src/evaluate_checkpoints.py results/mnist/<run> [<run> ...] [--taus 0 0.025 0.1]
Writes <run>/metrics_recomputed.csv (the original metrics.csv is left untouched).
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd
import torch
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mnist_cnn import MnistCNN, evaluate, get_device, plasticity_metrics  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="run directories containing a checkpoints/ folder")
    parser.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.025, 0.1])
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    device = get_device()
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=tf)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)
    # Same fixed probe batch as src/mnist_cnn.py
    probe = torch.stack([test_set[i][0] for i in range(1024)]).to(device)
    probe_y = torch.tensor([test_set[i][1] for i in range(1024)]).to(device)

    model = MnistCNN().to(device)
    for run in args.runs:
        paths = glob.glob(os.path.join(run, "checkpoints", "checkpoint_epoch_*.pt"))
        paths.sort(key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))
        rows = []
        for path in paths:
            ckpt = torch.load(path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            test_loss, test_acc = evaluate(model, test_loader, device)
            rows.append({"epoch": ckpt["epoch"], "test_loss": test_loss, "test_acc": test_acc,
                         **plasticity_metrics(model, probe, probe_y, tuple(args.taus))})
        out = os.path.join(run, "metrics_recomputed.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{run}: {len(rows)} checkpoints -> {out}", flush=True)


if __name__ == "__main__":
    main()
