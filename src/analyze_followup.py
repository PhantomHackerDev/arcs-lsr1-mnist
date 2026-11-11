"""Follow-up analysis on the deep net (784-512-256-128-10).

Q1  one shared mu vs a mu_l per layer (Aditya's per-layer ratio test)
Q2  can Hessian momentum B~ = a B~ + (1-a) B_new replace history pairs?

  python src/analyze_followup.py  -> results/summary_followup.md, figures/fig5..7
"""
import glob
import json
import math
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from analyze import FIG, RES  # noqa: E402  (also applies the shared rcParams)

LAYER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4
LAYER_NAMES = ["layer 1 (784→512)", "layer 2 (512→256)", "layer 3 (256→128)", "layer 4 (128→10)"]
CFG_COLORS = {
    "deep_full_same": "#2a78d6", "deep_layer_same": "#eb6834", "deep_layermu_same": "#1baf7a",
    "deep_full_cross": "#2a78d6", "deep_layer_cross": "#eb6834", "deep_layermu_cross": "#1baf7a",
    "deep_sgd": "#8a8985", "deep_adam": "#3d3c39",
}
LABEL = {
    "deep_sgd": "SGD+momentum (lr 0.1)",
    "deep_adam": "Adam (lr 1e-3)",
    "deep_full_same": "full network, one μ",
    "deep_layer_same": "per-layer models, shared μ",
    "deep_layermu_same": "per-layer models, μ_l per layer",
    "deep_full_cross": "full network, one μ",
    "deep_layer_cross": "per-layer models, shared μ",
    "deep_layermu_cross": "per-layer models, μ_l per layer",
    "deep_layermu_floor1_same": "μ_l per layer + floor μ_l ≥ μ_g (round 3)",
    "deep_layermu_floor1_cross": "μ_l per layer + floor μ_l ≥ μ_g (round 3)",
    "deep_layermu_next_same": "μ_l per layer, ρ_l on next batch (round 3)",
    "deep_layermu_next_cross": "μ_l per layer, ρ_l on next batch (round 3)",
}
Q1 = ["deep_sgd", "deep_adam", "deep_full_same", "deep_layer_same", "deep_layermu_same",
      "deep_layermu_floor1_same", "deep_layermu_next_same", "deep_full_cross", "deep_layer_cross",
      "deep_layermu_cross", "deep_layermu_floor1_cross", "deep_layermu_next_cross"]


def load():
    runs = defaultdict(list)
    for f in glob.glob(os.path.join(RES, "deep_*.json")):
        m = re.match(r"(.+)_bs(\d+)_s(\d+)\.json$", os.path.basename(f))
        if m:
            with open(f) as fh:
                runs[(m.group(1), int(m.group(2)))].append(json.load(fh))
    return runs


def memory_of(cfg):
    """(history m, weight on the OLD model (hess_beta), rank cap, n-vectors stored per layer).
    The weight on B_new is 1 - hess_beta."""
    m = re.search(r"_m(\d+)", cfg)
    m = int(m.group(1)) if m else 10
    hb = re.search(r"_hb([\d.]+?)(?:_|$)", cfg)
    hb = float(hb.group(1)) if hb else 0.0
    c = re.search(r"_cap(\d+)", cfg)
    cap = int(c.group(1)) if c else (2 * m if hb > 0 else 0)
    return m, hb, cap, 2 * m + cap


def stats(logs):
    acc = np.array([L["epochs"][-1]["test_acc"] for L in logs])
    out = {"n": len(logs), "acc": acc.mean(), "acc_sd": acc.std(ddof=1) if len(acc) > 1 else 0.0,
           "train_loss": np.mean([L["epochs"][-1]["train_loss"] for L in logs]),
           "time": np.mean([L["epochs"][-1]["time"] for L in logs])}
    it0 = logs[0]["iters"]
    if it0 and "mu" in it0[0]:
        out["accept"] = np.mean([np.mean([d["acc"] for d in L["iters"]]) for L in logs])
        out["fallback"] = np.mean([np.mean([d.get("fallback", 0) for d in L["iters"]]) for L in logs])
        mub = [L["epochs"][-1].get("mu_b") for L in logs if L["epochs"][-1].get("mu_b")]
        out["mu_b"] = list(np.median(np.array(mub), 0)) if mub else None
        out["mu_max"] = np.median([max(d["mu"] for d in L["iters"]) for L in logs])
    mv = [L["epochs"][-1].get("move_b") for L in logs if L["epochs"][-1].get("move_b")]
    out["move_b"] = list(np.mean(np.array(mv), 0)) if mv else None
    return out


def fmt_mu(v):
    return "–" if v is None else " / ".join(f"{x:.2g}" for x in v)


def table(runs):
    lines = ["### Q1: one μ vs. μ_l per layer (deep net, 10 epochs, mean ± sd over seeds)", "",
             "| bs | y | method | test acc % | train loss | accept % | fallback % | final μ_l (layers 1/2/3/4, median) "
             "| max μ | weight moved ‖Δw‖/‖w₀‖ (layers 1/2/3/4) | time s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for bs in (64, 256):
        for c in Q1:
            if (c, bs) not in runs:
                continue
            s = stats(runs[(c, bs)])
            y = "same" if c.endswith("same") else ("cross" if c.endswith("cross") else "")
            lines.append("| " + " | ".join([
                str(bs), y, LABEL[c], f"{100 * s['acc']:.2f} ± {100 * s['acc_sd']:.2f}", f"{s['train_loss']:.3f}",
                f"{100 * s['accept']:.1f}" if "accept" in s else "", f"{100 * s['fallback']:.1f}" if "fallback" in s else "",
                fmt_mu(s.get("mu_b")) if "mu_b" in s else "", f"{s['mu_max']:.2g}" if "mu_max" in s else "",
                fmt_mu(s.get("move_b")), f"{s['time']:.0f}"]) + " |")
    lines += ["", "### Q2: history size vs. Hessian momentum (deep net, bs 256, per-layer μ_l, same-batch y)", "",
              "B̃ = w_old·B̃ + w_new·B_new. Rounds 2 used w_new = 0.5 / 0.1; round 4 flips it to w_new = 0.7 / 0.9 "
              "(Aditya's suggestion) and also tries rank cap m.", "",
              "| history m | weight on B_new | rank cap | n-vectors stored per layer | test acc % | train loss | time s |",
              "|---|---|---|---|---|---|---|"]
    mem = sorted({c for (c, bs) in runs if c.startswith("deep_layermu_same") and bs == 256
                  and "floor" not in c and "next" not in c},
                 key=lambda c: (memory_of(c)[3], memory_of(c)[1]))
    for c in mem:
        m, hb, cap, vec = memory_of(c)
        s = stats(runs[(c, 256)])
        lines.append(f"| {m} | {f'{1 - hb:.1f}' if hb else '– (plain L-SR1)'} | {cap if cap else '–'} | {vec} | "
                     f"{100 * s['acc']:.2f} ± {100 * s['acc_sd']:.2f} | {s['train_loss']:.3f} | {s['time']:.0f} |")
    for b in ("deep_adam", "deep_sgd"):
        if (b, 256) in runs:
            s = stats(runs[(b, 256)])
            lines.append(f"| {LABEL[b]} | | | 0–2 | {100 * s['acc']:.2f} ± {100 * s['acc_sd']:.2f} | "
                         f"{s['train_loss']:.3f} | {s['time']:.0f} |")
    lines += ["", "Wall-clock times (time s) are not comparable between rows: up to 8 runs shared 16 cores "
              "with varying load. Only order-of-magnitude differences (first-order vs. ARCs) are meaningful."]
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(RES, "summary_followup.md"), "w") as f:
        f.write(txt)
    return txt


def secant_table(runs):
    """Newest-pair secant residual ||B s_k - y_k|| / ||y_k|| of the model used for the step."""
    rows = []
    for (c, bs), logs in runs.items():
        if not c.startswith("deep_secant_m2_hb") or bs != 256:
            continue
        hb = float(c.split("_hb")[1])
        L = logs[0]
        res = np.array([np.nanmean(d["sec_res"]) for d in L["iters"] if "sec_res" in d and d["k_pairs"][0] > 0])
        rows.append((1 - hb, np.nanmedian(res), np.nanmean(res), L["epochs"][-1]["test_acc"]))
    if not rows:
        return ""
    rows.sort()
    lines = ["", "### Secant residual vs. weight on B_new (m = 2, rank cap 4, seed 0, 3 epochs, batch 256)", "",
             "‖B̃ s_k − y_k‖ / ‖y_k‖ for the newest pair, averaged over layers. With weight w on B_new, "
             "B̃ s_k − y_k = (1 − w)(B̃_old s_k − y_k), so the residual should shrink linearly as w → 1.", "",
             "| weight on B_new | median residual | mean residual | test acc % after 3 epochs |", "|---|---|---|---|"]
    for w, med, mean, acc in rows:
        label = "1.0 (plain L-SR1)" if w == 1.0 else f"{w:.1f}"
        lines.append(f"| {label} | {med:.3g} | {mean:.3g} | {100 * acc:.2f} |")
    return "\n".join(lines) + "\n"


def fig_mu_layers(runs):
    panels = [("deep_layermu_same", 64), ("deep_layermu_same", 256), ("deep_layermu_cross", 64),
              ("deep_layermu_cross", 256)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
    for ax, (c, bs) in zip(axes, panels):
        logs = runs.get((c, bs))
        if logs:
            L = sorted(logs, key=lambda r: r["args"]["seed"])[0]
            mub = np.array([d["mu_b"] for d in L["iters"]])
            w = max(5, len(mub) // 150)
            k = np.ones(w) / w
            for j in range(mub.shape[1]):
                y = 10 ** np.convolve(np.log10(np.maximum(mub[:, j], 1e-12)), k, mode="valid")
                ax.plot(np.arange(len(y)) + w // 2, y, color=LAYER_COLORS[j], lw=1.6, label=LAYER_NAMES[j])
        ax.set_yscale("log")
        ax.set_title(f"{'same-batch' if c.endswith('same') else 'cross-batch'} y, batch {bs}")
        ax.set_xlabel("iteration")
    axes[0].set_ylabel("μ_l (log, smoothed)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Per-layer cubic regularization μ_l (seed 0): where does each layer's model need damping?",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(os.path.join(FIG, "fig5_layer_mu.png"), dpi=150)
    plt.close(fig)


def fig_layer_movement(runs, bs=256):
    cfgs = ["deep_sgd", "deep_adam", "deep_full_same", "deep_layer_same", "deep_layermu_same"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(1, 5)
    for c in cfgs:
        logs = runs.get((c, bs), [])
        mv = [L["epochs"][-1]["move_b"] for L in logs if L["epochs"][-1].get("move_b")]
        if not mv:
            continue
        mv = np.array(mv)
        ls = "--" if c in ("deep_sgd", "deep_adam") else "-"
        ax.plot(x, mv.mean(0), ls, marker="o", ms=6, color=CFG_COLORS[c], label=LABEL[c])
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" (", "\n(") for n in LAYER_NAMES])
    ax.set_ylabel("‖w_final − w_init‖ / ‖w_init‖")
    ax.set_title(f"How far each layer moved during training (batch {bs}, mean over seeds)", loc="left")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_layer_movement.png"), dpi=150)
    plt.close(fig)


def fig_memory(runs):
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    plain, mom = [], defaultdict(list)
    for (c, bs), logs in runs.items():
        if bs != 256 or not c.startswith("deep_layermu_same") or "floor" in c or "next" in c:
            continue
        m, hb, cap, vec = memory_of(c)
        s = stats(logs)
        pt = (vec, 100 * (1 - s["acc"]), 100 * s["acc_sd"], m)
        (mom[hb] if hb else plain).append(pt)
    if plain:
        plain.sort()
        a = np.array(plain)
        ax.errorbar(a[:, 0], a[:, 1], yerr=a[:, 2], fmt="-o", color="#2a78d6", capsize=3, ms=6,
                    label="L-SR1 history only (m = 1, 2, 5, 10)")
        for v, e, _, m in plain:
            ax.annotate(f"m={int(m)}", (v, e), textcoords="offset points", xytext=(6, 6), fontsize=8.5,
                        color="#52514e")
    # weight on B_new = 1 - hess_beta; offsets keep error bars at equal memory apart
    for hb, color, mk, dodge in ((0.9, "#1baf7a", "D", 0.91), (0.5, "#eb6834", "s", 0.97),
                                 (0.3, "#e87ba4", "^", 1.03), (0.1, "#6250d6", "v", 1.09)):
        if mom.get(hb):
            a = np.array(sorted(mom[hb]))
            ax.errorbar(a[:, 0] * dodge, a[:, 1], yerr=a[:, 2], fmt=mk, color=color, capsize=3, ms=7,
                        label=f"+ Hessian momentum, weight on B_new = {1 - hb:.1f}")
    for b in ("deep_sgd", "deep_adam"):
        if (b, 256) in runs:
            ax.axhline(100 * (1 - stats(runs[(b, 256)])["acc"]), color=CFG_COLORS[b], ls="--", lw=1.2,
                       label=LABEL[b])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([2, 3, 4, 6, 8, 10, 20])
    ax.set_ylim(1.4, 2.1)
    ax.set_xlabel("n-vectors stored per layer  (2m for pairs, + rank cap for momentum)")
    ax.set_ylabel("test error after 10 epochs (%)")
    ax.set_title("Memory vs. accuracy (deep net, batch 256, per-layer μ_l, mean ± sd)", loc="left")
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, fontsize=8.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(os.path.join(FIG, "fig7_memory.png"), dpi=150)
    plt.close(fig)


def main():
    runs = load()
    txt = table(runs) + secant_table(runs)
    with open(os.path.join(RES, "summary_followup.md"), "w") as f:
        f.write(txt)
    print(txt)
    fig_mu_layers(runs)
    fig_layer_movement(runs)
    fig_memory(runs)


if __name__ == "__main__":
    main()
