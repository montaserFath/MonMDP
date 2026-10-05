"""Direct plasticity probe: fine-tune every saved checkpoint on one new shuffled-MNIST task and compare the
accuracy reached with a freshly initialised network (same budget). Also computes the proxy metrics on each
checkpoint (on probe-task images) and reports their Spearman correlation with the probe accuracy.

Usage: python src/probe_plasticity.py results/mnist/<run> [<run> ...] [--task label|pixel] [--every 5]
Writes <run>/probe.csv per run and <out_dir>/probe_correlations.csv.
Checkpoints may be epoch checkpoints (src/mnist_cnn.py) or task checkpoints (src/continual_mnist.py).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from torch.nn import functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mnist_cnn import build_model, get_device, plasticity_metrics  # noqa: E402
from src.tasks import apply_task, evaluate_tensors, load_mnist, make_task  # noqa: E402

PROXIES = ["dormant_pct_tau_0", "dormant_pct_tau_0.025", "dormant_pct_tau_0.1", "srank", "effective_rank",
           "unit_sign_entropy", "grad_norm", "weight_norm", "active_params_pct"]


def finetune(model, data, args, device, seed) -> dict:
    """Fresh Adam, fixed batch order (seeded); returns probe-task test accuracy after args.early_steps and at the end."""
    xt, yt, xe, ye = data
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    out, step = {}, 0
    while step < args.steps:
        order = torch.randperm(len(xt), generator=gen).to(device)
        for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
            idx = order[i:i + args.batch_size]
            loss = F.cross_entropy(model(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step == args.early_steps:
                out["acc_early"] = evaluate_tensors(model, xe, ye)[1]
            if step >= args.steps:
                break
    out["loss_final"], out["acc_final"] = evaluate_tensors(model, xe, ye)
    return out


def list_checkpoints(run: str) -> list:
    paths = glob.glob(os.path.join(run, "checkpoints", "checkpoint_*_*.pt"))
    return sorted(paths, key=lambda p: int(re.search(r"_(\d+)\.pt$", p).group(1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--task", choices=["pixel", "label"], default="label")
    parser.add_argument("--probe_seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=938, help="fine-tuning steps (938 = 1 epoch at batch 64)")
    parser.add_argument("--early_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_fresh", type=int, default=3, help="fresh-init baselines to average")
    parser.add_argument("--every", type=int, default=1, help="use every k-th checkpoint")
    parser.add_argument("--probe_size", type=int, default=1024)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.025, 0.1])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="results/probe")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = get_device()
    x_train, y_train, x_test, y_test = load_mnist(args.data_dir, device)
    task = make_task(args.task, args.probe_seed, 1)  # non-identity probe task
    xt, yt = apply_task(x_train, y_train, task)
    xe, ye = apply_task(x_test, y_test, task)
    data = (xt, yt, xe, ye)
    probe_x, probe_y = xt[:args.probe_size], yt[:args.probe_size]

    all_rows = []
    for run in args.runs:
        try:
            with open(os.path.join(run, "args.json")) as f:
                arch = json.load(f).get("arch", "cnn")
        except FileNotFoundError:
            arch = "cnn"
        # Fresh-init baseline: same budget from random initialisation
        fresh = []
        for s in range(args.n_fresh):
            torch.manual_seed(10_000 + s)
            fresh.append(finetune(build_model(arch).to(device), data, args, device, seed=args.probe_seed))
        base_early = float(np.mean([r["acc_early"] for r in fresh]))
        base_final = float(np.mean([r["acc_final"] for r in fresh]))
        print(f"{run}: fresh-init baseline acc_early={base_early:.2f} acc_final={base_final:.2f}", flush=True)

        rows = []
        for path in list_checkpoints(run)[::args.every]:
            ckpt = torch.load(path, map_location=device)
            model = build_model(arch).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            proxies = plasticity_metrics(model, probe_x, probe_y, tuple(args.taus))  # before fine-tuning
            res = finetune(model, data, args, device, seed=args.probe_seed)
            idx = ckpt.get("epoch", ckpt.get("task"))
            rows.append({"index": idx, **res, "baseline_acc_early": base_early, "baseline_acc_final": base_final,
                         "gap_early": base_early - res["acc_early"], "gap_final": base_final - res["acc_final"],
                         **{k: v for k, v in proxies.items() if k in PROXIES}})
            print(f"  ckpt {idx}: acc_early={res['acc_early']:.2f} acc_final={res['acc_final']:.2f}", flush=True)
        df = pd.DataFrame(rows)
        df.insert(0, "run", os.path.basename(run.rstrip("/")))
        df.to_csv(os.path.join(run, "probe.csv"), index=False)
        all_rows.append(df)

    big = pd.concat(all_rows, ignore_index=True)
    corr = []
    for proxy in [p for p in PROXIES if p in big]:
        for target in ("acc_early", "acc_final"):
            rho, pval = st.spearmanr(big[proxy], big[target])
            corr.append({"proxy": proxy, "target": target, "spearman": rho, "p_value": pval, "n": len(big)})
    corr = pd.DataFrame(corr)
    corr.to_csv(os.path.join(args.out_dir, "probe_correlations.csv"), index=False)
    print(corr.pivot(index="proxy", columns="target", values="spearman").round(2).to_string())
    print("Note: pooled over checkpoints, so every proxy that trends with training time will correlate; "
          "compare within-run for a stricter test.")


if __name__ == "__main__":
    main()
