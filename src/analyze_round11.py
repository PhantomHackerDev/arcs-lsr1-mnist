"""Round 11: the paper's delta rule (delta = lambda_hat_min) vs. 0.9 and 0.5 * lambda_hat_min, F-MNIST CNN.

  python src/analyze_round11.py -> results/summary_round11.md, figures/fig16_delta_rule.png
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from analyze import FIG, RES  # noqa: E402  (shared rcParams)
from analyze_round9 import COLORS, NAMES, at, base_name, curve, fmt, lam_neg, load  # noqa: E402
from sweep_round9m import name as name9  # noqa: E402
from sweep_round10 import name as name10  # noqa: E402
from sweep_round11 import name as name11  # noqa: E402

with open(os.path.join(RES, "round9_selection.json")) as f:
    SEL = json.load(f)
BSS, MS = (1024, 2048, 4096), (10, 50, 100)


def ref_logs(bs, m):
    """delta = 0.5 * lambda_hat_min with the LM-damped handling of lambda_hat_min <= 0 ('safe_lm')"""
    if bs == 2048:
        return load(name11(SEL, bs, m, "lm0.5") if m == 10 else name10(SEL, bs, m, "lm"), bs)
    logs = load(SEL[f"fm_bs{bs}_arcs_pairs"]["name"], bs) if m == 10 else load(name9(SEL, bs, m, True), bs)
    return logs if (logs and lam_neg(logs) == 0) else []  # identical to 'safe_lm' only if lambda_hat_min > 0


def at_safe(logs, ep, key="train_loss"):
    """mean, sd, n at epoch ep; a run that stopped early (non-finite loss) counts as diverged (inf / 0)"""
    bad = np.inf if key == "train_loss" else 0.0
    v = np.array([curve(L, key)[ep] if len(L["epochs"]) > ep else bad for L in logs], dtype=float)
    v = np.where(np.isfinite(v), v, bad)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else None), len(v)


def stat(logs, key, f=np.median):
    v = np.array([np.ravel(d[key])[0] for L in logs for d in L["iters"] if key in d], dtype=float)
    v = v[np.isfinite(v)]
    return f(v) if len(v) else float("nan")


RULES = [("paper", "δ = λ̂_min (paper, Alg. 1)", "#eb6834", "s"),
         ("lm0.9", "δ = 0.9 λ̂_min, LM damping if λ̂_min ≤ 0", "#6250d6", "^"),
         ("lm0.5", "δ = 0.5 λ̂_min, LM damping if λ̂_min ≤ 0 (rounds 9-10)", "#2a78d6", "o"),
         ("lm0.25", "δ = 0.25 λ̂_min, LM damping if λ̂_min ≤ 0", "#1baf7a", "v"),
         ("lm0.1", "δ = 0.1 λ̂_min, LM damping if λ̂_min ≤ 0", "#eda100", "D")]


def main():
    lines = ["### Round 11: the paper's δ rule in the current setting (F-MNIST CNN, 10 epochs, 3 seeds)", "",
             "Same-batch y, full network, (η₂, μ_min) of round 9 per batch size. 'paper': δ = λ̂_min, the smallest "
             "eigenvalue of (D + L + Lᵀ)u = λ̂ SᵀS u, clipped to [1e-4, 1e4] (so δ = 1e-4 when λ̂_min ≤ 1e-4). "
             "cond(M⁻¹) is the median over steps.", "",
             "| batch | m | δ rule | train loss | test acc % | accepted % | median δ | δ at the 1e-4 clip (% of steps) | "
             "λ̂_min ≤ 0 (% of steps) | median cond(M⁻¹) |", "|---|---|---|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, len(BSS), figsize=(17, 4.8))
    for ax, bs in zip(axes, BSS):
        for rule, lab, c, mk in RULES:
            pts = []
            for m in MS:
                logs = ref_logs(bs, m) if rule == "lm0.5" else load(name11(SEL, bs, m, rule), bs)
                if not logs:
                    continue
                tl, ta = at_safe(logs, 10), at_safe(logs, 10, "test_acc")
                if not np.isfinite(tl[0]):
                    tl = (tl[0], None, tl[2])
                clip = 100 * stat(logs, "delta", lambda v: np.mean(v <= 1.0001e-4))
                lines.append(f"| {bs} | {m} | {lab} | {fmt(*tl)} | {100 * ta[0]:.2f} | {100 * stat(logs, 'acc', np.mean):.0f} | "
                             f"{stat(logs, 'delta'):.3g} | {clip:.0f} | {lam_neg(logs):.0f} | {stat(logs, 'cond_minv'):.2g} |")
                v = np.array([curve(L)[10] if len(curve(L)) > 10 else np.inf for L in logs])
                v = np.where(np.isfinite(v), v, np.nan)
                pts.append((m, np.nanmean(v), np.nanmin(v), np.nanmax(v)))
            if pts:
                m_, y, lo_, hi_ = map(np.array, zip(*pts))
                ax.errorbar(m_, y, yerr=[y - lo_, hi_ - y], color=c, marker=mk, ms=5, lw=1.8, capsize=3, label=lab)
        for opt, ls in (("adamw", "-."), ("shampoo", ":"), ("muon", "--")):
            b = load(base_name(opt, SEL[f"fm_bs{bs}_{opt}"]["lr_epoch10"]), bs)
            if b:
                ax.axhline(at(b, 10)[0], color=COLORS[opt], ls=ls, lw=1.1, label=f"{NAMES[opt]} (tuned, epoch 10)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(list(MS))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=(2, 3, 4, 5, 6)))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.tick_params(axis="y", which="minor", labelsize=7)
        ax.set_xlabel("L-SR1 memory m")
        ax.set_title(f"batch {bs}")
        lines.append("")
    axes[0].set_ylabel("training loss after 10 epochs (log)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("F-MNIST CNN: the δ rule of ARCs-LSR1 (mean over 3 seeds, bars = min-max)", x=0.01, ha="left",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(os.path.join(FIG, "fig16_delta_rule.png"), dpi=150)
    plt.close(fig)
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(RES, "summary_round11.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
