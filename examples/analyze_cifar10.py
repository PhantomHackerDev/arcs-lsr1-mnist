"""Summary of the CIFAR-10 sweep (examples/sweep_cifar10.py).

  python examples/analyze_cifar10.py            -> results/cifar10/summary.md, results/cifar10/fig_cifar10.png
  python examples/analyze_cifar10.py --smoke    -> the same for results/cifar10_smoke/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from sweep_cifar10 import Sweep  # noqa: E402

NAMES = {"adamw": "AdamW", "sgd": "SGD+momentum", "muon": "Muon"}
COLORS = {"arcs_diag0": "#2a78d6", "arcs_diag1": "#1baf7a", "adamw": "#6250d6", "sgd": "#8a8985", "muon": "#3d3c39"}


def load(out, tag_s0):
    logs = []
    for f in sorted(glob.glob(os.path.join(out, tag_s0[:-1] + "[0-9].json"))):
        with open(f) as fh:
            logs.append(json.load(fh))
    return logs


def at(logs, ep, key="train_loss"):
    v = np.array([L["epochs"][ep][key] if len(L["epochs"]) > ep else np.inf for L in logs], dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else None), len(v)


def fmt(m, sd, n, p=4):
    return f"{m:.{p}f}" + (f" ± {sd:.{p}f}" if sd is not None else "") + (f" (n={n})" if n > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or os.path.join(ROOT, "results", "cifar10_smoke" if a.smoke else "cifar10")
    a.epochs, a.dry_run, a.gpus, a.per_gpu, a.threads = 0, True, "cpu", 1, 1
    sw = Sweep(a)
    with open(os.path.join(a.out, "selection.json")) as f:
        sel = json.load(f)
    E = sel["E"]
    rows = []  # (key, label, logs at epoch E, logs at 2E)
    for p in (0, 1):
        k = f"arcs_diag{p}"
        if k in sel:
            s = sel[k]
            logs = load(a.out, sw.tag_arcs(s["m"], s["mu_min"], s["diag_power"], 0))
            lab = f"ARCs-LSR1{', diagonal B0' if p else ''} (m = {s['m']}, μ_min = {s['mu_min']:g})"
            rows.append((k, lab, logs, logs))
    for o in ("adamw", "sgd", "muon"):
        if o in sel:
            s = sel[o]
            lE, l2E = load(a.out, sw.tag_base(o, s["lr_epochE"], 0)), load(a.out, sw.tag_base(o, s["lr_epoch2E"], 0))
            lr = f"lr {s['lr_epochE']:.3g}" + ("" if s["lr_epoch2E"] == s["lr_epochE"] else f" / {s['lr_epoch2E']:.3g}")
            rows.append((o, f"{NAMES[o]} ({lr}{', edge' if s['at_edge'] else ''})", lE, l2E))

    first = rows[0][2][0] if rows and rows[0][2] else None
    lines = [f"### CIFAR-10 / ResNet-18, batch {a.bs}{', augmentation' if a.augment else ''}", "",
             f"Training loss on the full training set (eval mode) and test accuracy, mean ± sd over seeds. 'Epoch {E}': "
             f"same number of samples and steps. 'Same compute': ARCs-LSR1 after {E} epochs vs. the baselines after "
             f"{2 * E} epochs (their learning rate chosen at that epoch).",
             (f"Hardware: {first['gpu'] or first['device']}, torch {first['torch']}." if first else ""), "",
             f"| method | train loss, epoch {E} | test acc %, epoch {E} | train loss, same compute | "
             f"test acc %, same compute | wall time per epoch (s) | max GPU memory (GB) |",
             "|---|---|---|---|---|---|---|"]
    for k, lab, lE, l2E in rows:
        if not lE:
            continue
        ep2 = E if k.startswith("arcs") else 2 * E
        tpe = np.mean([L["epochs"][-1]["time"] / max(L["epochs"][-1]["epoch"], 1) for L in lE])
        mem = [L["epochs"][-1].get("max_mem_gb") for L in lE]
        mem = f"{np.mean(mem):.1f}" if all(m is not None for m in mem) else "–"
        lines.append(f"| {lab} | {fmt(*at(lE, E))} | {100 * at(lE, E, 'test_acc')[0]:.2f} | {fmt(*at(l2E, ep2))} | "
                     f"{100 * at(l2E, ep2, 'test_acc')[0]:.2f} | {tpe:.0f} | {mem} |")
    lines += ["", "Seed-0 grids (training loss):", ""]
    for o in ("adamw", "sgd", "muon"):
        if o in sel:
            lines.append(f"- {NAMES[o]}: " + ", ".join(f"lr {lr}: {v['epochE']:.4f} (ep {E}) / {v['epoch2E']:.4f} "
                                                      f"(ep {2 * E})" for lr, v in sel[o]["tried"].items()))
    for p in (0, 1):
        k = f"arcs_diag{p}"
        if k in sel:
            lines.append(f"- ARCs-LSR1{' diagonal B0' if p else ''} (epoch {E}): " +
                         ", ".join(f"{c}: {v:.4f}" for c, v in sel[k]["tried"].items()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for k, lab, lE, l2E in rows:
        if not lE:
            continue
        c, ls = COLORS[k], "-" if k.startswith("arcs") else "--"
        M = np.array([[e["train_loss"] for e in L["epochs"][1:E + 1]] for L in lE])
        axes[0].plot(np.arange(1, M.shape[1] + 1), np.exp(np.log(M).mean(0)), ls, color=c, marker="o", ms=3, lw=1.8,
                     label=f"{lab} (n={len(lE)})")
        n = min(len(L["epochs"]) for L in l2E)
        M = np.array([[e["train_loss"] for e in L["epochs"][1:n]] for L in l2E])
        X = np.array([[e["grad_epochs"] for e in L["epochs"][1:n]] for L in l2E]).mean(0)
        axes[1].plot(X, np.exp(np.log(M).mean(0)), ls, color=c, marker="o", ms=3, lw=1.8)
    for ax, xl in zip(axes, ("epoch", "compute (full-training-set gradient evaluations)")):
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.set_xlabel(xl)
        ax.set_ylabel("training loss (full training set, eval mode, log)")
    axes[0].set_title(f"per epoch (first {E} epochs)")
    axes[1].set_title("per unit of compute")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"CIFAR-10 / ResNet-18, batch {a.bs}: ARCs-LSR1 vs. tuned baselines (geometric mean over seeds)",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_cifar10.png"), dpi=150)
    plt.close(fig)
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
