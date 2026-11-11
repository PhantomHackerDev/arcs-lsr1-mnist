"""Round 5: the paper's F-MNIST CNN (21,840 params, 4 layer blocks).

  python src/analyze_fmnist.py  -> results/summary_fmnist.md, figures/fig8_fmnist.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from analyze import FIG, RES  # noqa: E402  (shared rcParams)
from analyze_followup import load as _load  # noqa: E402,F401
import glob  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from collections import defaultdict  # noqa: E402

ORDER = ["fm_sgd", "fm_adam", "fm_full_same", "fm_layer_same", "fm_layermu_floor_same", "fm_layer_same_m2",
         "fm_layermu_floor_same_m2", "fm_full_cross", "fm_layer_cross", "fm_layermu_floor_cross"]
LABEL = {
    "fm_sgd": "SGD+momentum (lr 0.1)", "fm_adam": "Adam (lr 1e-3)",
    "fm_full_same": "ARCs-LSR1, full network, m=10", "fm_layer_same": "per-layer models, shared μ, m=10",
    "fm_layermu_floor_same": "per-layer μ_l + floor, m=10", "fm_layer_same_m2": "per-layer models, shared μ, m=2",
    "fm_layermu_floor_same_m2": "per-layer μ_l + floor, m=2",
    "fm_full_cross": "full network, m=10", "fm_layer_cross": "per-layer models, shared μ, m=10",
    "fm_layermu_floor_cross": "per-layer μ_l + floor, m=10",
}
COLOR = {"fm_sgd": "#8a8985", "fm_adam": "#3d3c39", "fm_full_same": "#2a78d6", "fm_layer_same": "#eb6834",
         "fm_layermu_floor_same": "#1baf7a", "fm_layer_same_m2": "#eda100", "fm_layermu_floor_same_m2": "#e87ba4"}


def load():
    runs = defaultdict(list)
    for f in glob.glob(os.path.join(RES, "fm_*.json")):
        m = re.match(r"(.+)_bs(\d+)_s(\d+)\.json$", os.path.basename(f))
        if m:
            with open(f) as fh:
                runs[(m.group(1), int(m.group(2)))].append(json.load(fh))
    return runs


def row(logs):
    acc = np.array([L["epochs"][-1]["test_acc"] for L in logs]) * 100
    best = np.array([max(e["test_acc"] for e in L["epochs"]) for L in logs]) * 100
    out = {"acc": acc.mean(), "sd": acc.std(ddof=1) if len(acc) > 1 else 0.0, "best": best.mean(),
           "loss": np.mean([L["epochs"][-1]["train_loss"] for L in logs]), "n": len(logs),
           "seeds": acc}
    it = logs[0]["iters"]
    if it and "mu" in it[0]:
        out["accept"] = 100 * np.mean([np.mean([d["acc"] for d in L["iters"]]) for L in logs])
        out["cap"] = 100 * np.mean([np.mean([max(d["mu_b"]) >= 1e11 for d in L["iters"]]) for L in logs])
    return out


def table(runs):
    lines = ["### Round 5: F-MNIST, the paper's CNN (10 epochs, mean ± sd over seeds)", "",
             "| bs | y | method | test acc % (final) | best-epoch acc % | train loss | accept % | iterations with a μ at cap % |",
             "|---|---|---|---|---|---|---|---|"]
    for bs in (64, 256):
        for c in ORDER:
            if (c, bs) not in runs:
                continue
            r = row(runs[(c, bs)])
            y = "same" if c.endswith(("same", "same_m2")) else ("cross" if c.endswith("cross") else "")
            acc_s = f"{r['accept']:.1f}" if "accept" in r else ""
            cap_s = f"{r['cap']:.1f}" if "cap" in r else ""
            lines.append(f"| {bs} | {y} | {LABEL[c]} | {r['acc']:.2f} ± {r['sd']:.2f} | {r['best']:.2f} | "
                         f"{r['loss']:.3f} | {acc_s} | {cap_s} |")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(RES, "summary_fmnist.md"), "w") as f:
        f.write(txt)
    return txt


def fig(runs):
    cfgs = ["fm_sgd", "fm_adam", "fm_full_same", "fm_layer_same", "fm_layermu_floor_same", "fm_layer_same_m2",
            "fm_layermu_floor_same_m2"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
    for col, bs in enumerate((64, 256)):
        for row, (key, scale) in enumerate((("train_loss", 1.0), ("test_acc", 100.0))):
            ax = axes[row, col]
            for c in cfgs:
                logs = runs.get((c, bs))
                if not logs:
                    continue
                v = np.array([[e[key] for e in L["epochs"]] for L in logs]) * scale
                ep = np.arange(v.shape[1])
                ls = "--" if c in ("fm_sgd", "fm_adam") else "-"
                ax.plot(ep[1:], v.mean(0)[1:], ls, marker="o", ms=3.5, color=COLOR[c], label=LABEL[c])
                ax.fill_between(ep[1:], v.min(0)[1:], v.max(0)[1:], color=COLOR[c], alpha=0.12, lw=0)
            if row == 0:
                ax.set_title(f"batch size {bs}")
                ax.set_yscale("log")
                ax.set_yticks([0.2, 0.3, 0.4, 0.6, 1.0])
                ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            else:
                ax.set_xlabel("epoch")
    axes[0, 0].set_ylabel("training loss (full training set, log)")
    axes[1, 0].set_ylabel("test accuracy (%)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005), fontsize=9)
    fig.suptitle("F-MNIST, the paper's CNN: same-batch ARCs-LSR1 variants vs. first-order baselines "
                 "(mean, band = min–max over 3 seeds)", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(os.path.join(FIG, "fig8_fmnist.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    runs = load()
    print(table(runs))
    fig(runs)
