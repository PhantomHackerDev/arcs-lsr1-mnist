"""Training response vs. the Hessian-momentum weight (Aditya: "measure the improvement of the
training response w.r.t. the alpha flip term at each batch").

Deep net 784-512-256-128-10, batch 256, per-layer mu_l, same-batch pairs, m = 2, 3 seeds.
B~ = (1 - w) B~_old + w B_new, w = weight on B_new (w = 1: plain L-SR1).

Per-batch improvement: the actual decrease of the mini-batch loss from the step taken,
  d_k = f_B(x_k) - f_B(x_{k+1}) = rho_k * pred_k  (0 if the step was rejected),
reported relative to f_B(x_k).

  python src/analyze_training.py -> results/summary_training.md, figures/fig9_training_alpha.png
"""
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze import FIG, RES  # noqa: E402  (shared rcParams)

# (config, label, color): the ordered w series uses one blue ramp, light -> dark
SERIES = [
    ("deep_layermu_same_m2_hb0.9", "m=2, w = 0.1", "#86b6ef"),
    ("deep_layermu_same_m2_hb0.5", "m=2, w = 0.5", "#3987e5"),
    ("deep_layermu_same_m2_hb0.3_cap4", "m=2, w = 0.7", "#256abf"),
    ("deep_layermu_same_m2_hb0.1_cap4", "m=2, w = 0.9", "#184f95"),
    ("deep_layermu_same_m2", "m=2, w = 1 (plain L-SR1)", "#0d366b"),
    ("deep_layermu_same", "m=10, plain L-SR1", "#eb6834"),
    ("deep_sgd", "SGD+momentum (lr 0.1)", "#8a8985"),
    ("deep_adam", "Adam (lr 1e-3)", "#3d3c39"),
]
# idle-machine ms per iteration (results/overhead.md, deep net, batch 256, 8 threads)
MS_PER_IT = {"deep_sgd": 3.4, "deep_adam": 4.2, "deep_layermu_same": 56.0,  # mu_l+floor m=10 (56.9 ms) minus the floor's extra forward
             "deep_layermu_same_m2": 31.9, "momentum": 44.1}


def ms_per_it(cfg):
    if cfg in MS_PER_IT:
        return MS_PER_IT[cfg]
    return MS_PER_IT["momentum"] if "_hb" in cfg else MS_PER_IT["deep_layermu_same_m2"]


def load(cfg, bs=256):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, f"{cfg}_bs{bs}_s*.json"))):
        if re.match(rf"{re.escape(cfg)}_bs{bs}_s\d+\.json$", os.path.basename(f)):
            with open(f) as fh:
                out.append(json.load(fh))
    return out


def per_batch_curves(logs):
    """mini-batch loss f_k, and relative per-batch decrease d_k / f_k, per iteration (mean over seeds)."""
    f = np.array([[d["loss"] for d in L["iters"]] for L in logs])
    rel = []
    for L in logs:
        r = []
        for d in L["iters"]:
            dec = d["rho"] * d["pred"] if d["acc"] and d["rho"] > -1e8 else 0.0
            r.append(dec / d["loss"] if d["loss"] > 0 else 0.0)
        rel.append(r)
    return f.mean(0), np.array(rel).mean(0)


def baseline_batch_loss(logs):
    """SGD/Adam log the mini-batch loss every 10 iterations."""
    it = np.array([d["it"] for d in logs[0]["iters"]])
    f = np.array([[d["loss"] for d in L["iters"]] for L in logs]).mean(0)
    return it, f


def smooth(v, w):
    v = np.asarray(v, float)
    if len(v) < w:
        return np.arange(len(v)), v
    k = np.ones(w) / w
    return np.arange(len(v) - w + 1) + w // 2, np.convolve(v, k, mode="valid")


def epochs_to(train_loss, thr):
    """first (linearly interpolated, in log loss) epoch at which the full training loss < thr."""
    tl = np.asarray(train_loss)
    for e in range(1, len(tl)):
        if tl[e] < thr:
            a, b = np.log(tl[e - 1]), np.log(tl[e])
            return e - 1 + (a - np.log(thr)) / (a - b) if a != b else e
    return float("nan")


def main():
    data = {cfg: load(cfg) for cfg, _, _ in SERIES}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    (a1, a2), (a3, a4) = axes
    rows = []
    for cfg, label, color in SERIES:
        logs = data[cfg]
        if not logs:
            continue
        ls = "--" if cfg in ("deep_sgd", "deep_adam") else "-"
        tl = np.array([[e["train_loss"] for e in L["epochs"]] for L in logs]).mean(0)
        ep = np.arange(len(tl))
        a1.plot(ep[1:], tl[1:], ls, marker="o", ms=3.5, color=color, label=label)
        n_it = logs[0]["epochs"][-1]["iter"]
        it_per_ep = n_it / (len(tl) - 1)
        # full training loss vs. estimated idle-machine wall time
        a4.plot(ep[1:] * it_per_ep * ms_per_it(cfg) / 1e3, tl[1:], ls, marker="o", ms=3.5, color=color, label=label)
        row = {"label": label, "n": len(logs), "tl": tl, "ms": ms_per_it(cfg)}
        if cfg in ("deep_sgd", "deep_adam"):
            it, f = baseline_batch_loss(logs)
            x, y = smooth(f, 5)
            a2.plot(it[x], y, ls, color=color, lw=1.4, label=label)
        else:
            f, rel = per_batch_curves(logs)
            x, y = smooth(f, 50)
            a2.plot(x, y, ls, color=color, lw=1.4, label=label)
            x, y = smooth(rel, 100)
            a3.plot(x, 100 * y, ls, color=color, lw=1.4, label=label)
            ip = int(round(it_per_ep))
            row["rel_ep1"] = 100 * rel[:ip].mean()
            row["rel_ep5"] = 100 * rel[4 * ip:5 * ip].mean()
            row["rel_ep10"] = 100 * rel[9 * ip:].mean()
            row["accept"] = 100 * np.mean([np.mean([d["acc"] for d in L["iters"]]) for L in logs])
        rows.append(row)
    a1.set_yscale("log")
    a1.set_xlabel("epoch")
    a1.set_ylabel("training loss (full training set)")
    a1.set_title("Training objective per epoch", loc="left")
    a2.set_yscale("log")
    a2.set_xlabel("iteration (batch)")
    a2.set_ylabel("mini-batch loss (moving average)")
    a2.set_title("Mini-batch loss at each batch", loc="left")
    a3.set_xlabel("iteration (batch)")
    a3.set_ylabel("decrease on the batch, % of f_B(x_k)\n(moving average over 100 batches)")
    a3.set_title("Per-batch improvement f_B(x_k) − f_B(x_k+1)  (ARCs runs only)", loc="left")
    a3.set_ylim(bottom=0)
    a4.set_yscale("log")
    a4.set_xlabel("estimated wall time on an idle machine (s)  [iterations × measured ms/iteration]")
    a4.set_ylabel("training loss (full training set)")
    a4.set_title("Training objective vs. time", loc="left")
    h, lab = a1.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005), fontsize=9)
    fig.suptitle("Training response vs. the weight w on B_new  (deep net, batch 256, per-layer μ_l, "
                 "same-batch pairs, mean over 3 seeds)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(os.path.join(FIG, "fig9_training_alpha.png"), dpi=150)
    plt.close(fig)

    lines = ["### Training response vs. the weight w on B_new (deep net, batch 256, per-layer μ_l, same-batch, "
             "mean over 3 seeds)", "",
             "Training loss is on the full 60k training set. The per-batch decrease is f_B(x_k) − f_B(x_k+1) "
             "on the batch the step was computed on, as % of f_B(x_k), averaged over the iterations of that epoch "
             "(rejected steps count as 0). Time per epoch uses the idle-machine benchmark (results/overhead.md).", "",
             "| method | train loss ep 1 | ep 3 | ep 10 | epochs to 0.05 | to 0.02 | to 0.01 | per-batch decrease "
             "ep 1 / 5 / 10 (%) | accept % | est. s / epoch |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tl = r["tl"]
        e = [epochs_to(tl, t) for t in (0.05, 0.02, 0.01)]
        dec = (f"{r['rel_ep1']:.1f} / {r['rel_ep5']:.1f} / {r['rel_ep10']:.1f}" if "rel_ep1" in r else "–")
        acc = f"{r['accept']:.0f}" if "accept" in r else "–"
        lines.append(f"| {r['label']} | {tl[1]:.4f} | {tl[3]:.4f} | {tl[10]:.4f} | "
                     + " | ".join("–" if np.isnan(v) else f"{v:.1f}" for v in e)
                     + f" | {dec} | {acc} | {234 * r['ms'] / 1e3:.1f} |")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(RES, "summary_training.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
