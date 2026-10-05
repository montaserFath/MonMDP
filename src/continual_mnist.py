"""Continual training on a sequence of shuffled-MNIST tasks while logging per-task accuracy and plasticity metrics.

A task is a fixed pixel permutation (--task pixel, as in Dohare et al. 2024) or a fixed class-id permutation
(--task label). One network and one optimizer are trained through the whole sequence. Plasticity metrics are
measured at the start of each task on 2000 images of the new task.
"""
import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd
import torch
from torch.nn import functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mnist_cnn import build_model, get_device, plasticity_metrics  # noqa: E402
from src.tasks import apply_task, evaluate_tensors, load_mnist, make_task  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["pixel", "label"], default="pixel")
    parser.add_argument("--arch", choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--n_tasks", type=int, default=100)
    parser.add_argument("--epochs_per_task", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.025, 0.1])
    parser.add_argument("--probe_size", type=int, default=2000)
    parser.add_argument("--ckpt_every", type=int, default=10)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="results/continual")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    run_dir = os.path.join(args.out_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = get_device()
    x_train, y_train, x_test, y_test = load_mnist(args.data_dir, device)
    model = build_model(args.arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = torch.Generator().manual_seed(args.seed)  # data order and probe sampling

    rows = []
    for t in range(args.n_tasks):
        task = make_task(args.task, args.seed, t)
        xt, yt = apply_task(x_train, y_train, task)
        xe, ye = apply_task(x_test, y_test, task)

        probe_idx = torch.randperm(len(xt), generator=rng)[:args.probe_size].to(device)
        metrics = plasticity_metrics(model, xt[probe_idx], yt[probe_idx], tuple(args.taus))

        model.train()
        correct, seen = 0, 0
        for _ in range(args.epochs_per_task):
            order = torch.randperm(len(xt), generator=rng).to(device)
            for i in range(0, len(order), args.batch_size):
                idx = order[i:i + args.batch_size]
                out = model(xt[idx])
                loss = F.cross_entropy(out, yt[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                correct += (out.argmax(1) == yt[idx]).sum().item()  # online accuracy: before the update
                seen += len(idx)
        test_loss, test_acc = evaluate_tensors(model, xe, ye)
        row = {"task": t, "online_acc": 100 * correct / seen, "test_loss": test_loss, "test_acc": test_acc, **metrics}
        rows.append(row)
        print(f"task={t} online_acc={row['online_acc']:.2f} test_acc={test_acc:.2f} dormant_pct={row['dormant_pct']:.1f} "
              f"srank={row['srank']} weight_norm={row['weight_norm']:.1f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

        if t % args.ckpt_every == 0 or t == args.n_tasks - 1:
            torch.save({"task": t, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "metrics": row},
                       os.path.join(ckpt_dir, f"checkpoint_task_{t}.pt"))
    torch.save(model.state_dict(), os.path.join(run_dir, "model_final.pt"))
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
