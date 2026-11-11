"""Round 6: modern baselines (AdamW, Muon, Shampoo, SophiaG; lr tuned by training loss on seed 0,
then 3 seeds) next to SGD/Adam (the paper's settings) and the best ARCs-LSR1 variants.

  python src/analyze_modern.py -> results/summary_modern.md, figures/fig10_modern_baselines.png
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from analyze import FIG, RES  # noqa: E402  (shared rcParams)

with open(os.path.join(RES, "round6_selection.json")) as f:
    SEL = json.load(f)

PANELS = [("deep", 256, "MNIST, deep MLP, batch 256"), ("fm", 64, "F-MNIST, CNN, batch 64"),
          ("fm", 256, "F-MNIST, CNN, batch 256")]
# (key, label, color, linestyle); ARCs first, then modern, then the paper's baselines
SERIES = [
    ("arcs_full", "ARCs-LSR1, full network, m=10", "#2a78d6", "-"),
    ("arcs_best", "ARCs-LSR1, per-layer μ_l (+ floor on F-MNIST), m=2", "#eb6834", "-"),
    ("adamw", "AdamW (tuned lr)", "#1baf7a", "-"),
    ("muon", "Muon (tuned lr)", "#eda100", "-"),
    ("shampoo", "Shampoo, Adam-grafted (tuned lr)", "#e87ba4", "-"),
    ("sophia", "SophiaG (tuned lr)", "#6250d6", "-"),
    ("sgd", "SGD+momentum (paper: lr 0.1)", "#8a8985", "--"),
    ("adam", "Adam (paper: lr 1e-3)", "#3d3c39", "--"),
]


def config(prefix, bs, key):
    """run-name stem (without _s{seed}) for a series in a panel."""
    if key in ("adamw", "muon", "shampoo", "sophia"):
        return f"{prefix}_{key}_lr{SEL[f'{prefix}_bs{bs}_{key}']['lr']:g}"
    if prefix == "deep":
        return {"arcs_full": "deep_full_same", "arcs_best": "deep_layermu_same_m2", "sgd": "deep_sgd",
                "adam": "deep_adam"}[key]
    return {"arcs_full": "fm_full_same", "arcs_best": "fm_layermu_floor_same_m2", "sgd": "fm_sgd",
            "adam": "fm_adam"}[key]


def load(stem, bs):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, f"{stem}_bs{bs}_s[0-9].json"))):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4), sharex=True)
    lines = ["### Round 6: modern baselines vs. ARCs-LSR1 (10 epochs, mean ± sd over 3 seeds)", "",
             "Modern baselines: learning rate chosen by the lowest final training loss on seed 0 over a sweep "
             "that was extended until the best value was inside the range (results/round6_selection.json); "
             "decoupled weight decay 0.01. SGD and Adam use the paper's settings.", ""]
    for col, (prefix, bs, title) in enumerate(PANELS):
        lines += [f"**{title}**", "", "| method | lr | test acc % | final train loss | train loss epoch 1 | epoch 3 |",
                  "|---|---|---|---|---|---|"]
        for key, label, color, ls in SERIES:
            stem = config(prefix, bs, key)
            logs = load(stem, bs)
            if not logs:
                continue
            tl = np.array([[e["train_loss"] for e in L["epochs"]] for L in logs])
            acc = np.array([[e["test_acc"] for e in L["epochs"]] for L in logs]) * 100
            ep = np.arange(tl.shape[1])
            for row, v in enumerate((tl, acc)):
                ax = axes[row, col]
                ax.plot(ep[1:], v.mean(0)[1:], ls, marker="o", ms=3, color=color, label=label, lw=1.8)
                ax.fill_between(ep[1:], v.min(0)[1:], v.max(0)[1:], color=color, alpha=0.10, lw=0)
            lr = (f"{SEL[f'{prefix}_bs{bs}_{key}']['lr']:.2g}" if key in ("adamw", "muon", "shampoo", "sophia")
                  else {"sgd": "0.1", "adam": "1e-3"}.get(key, "–"))
            lines.append(f"| {label} | {lr} | {acc[:, -1].mean():.2f} ± {acc[:, -1].std(ddof=1):.2f} | "
                         f"{tl[:, -1].mean():.4f} ± {tl[:, -1].std(ddof=1):.4f} | {tl[:, 1].mean():.4f} | "
                         f"{tl[:, 3].mean():.4f} |")
        lines.append("")
        axes[0, col].set_title(title)
        axes[0, col].set_yscale("log")
        if prefix == "fm":
            axes[0, col].set_yticks([0.15, 0.2, 0.3, 0.4, 0.6])
        axes[0, col].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        axes[0, col].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes[1, col].set_xlabel("epoch")
    axes[0, 0].set_ylabel("training loss (full training set, log)")
    axes[1, 0].set_ylabel("test accuracy (%)")
    axes[1, 0].set_ylim(96, 99)
    axes[1, 1].set_ylim(70, 93)
    axes[1, 2].set_ylim(76, 93)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005), fontsize=9)
    fig.suptitle("Modern baselines (lr tuned on training loss) vs. ARCs-LSR1 and the paper's SGD/Adam "
                 "(mean, band = min–max over 3 seeds)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(os.path.join(FIG, "fig10_modern_baselines.png"), dpi=150)
    plt.close(fig)
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(RES, "summary_modern.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
