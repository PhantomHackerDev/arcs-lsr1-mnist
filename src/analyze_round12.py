"""Round 12: diagonal initial matrix B0 = delta * D (diag_power p), F-MNIST CNN.

  python src/analyze_round12.py -> results/summary_round12.md, figures/fig17_diag_b0.png
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from analyze import FIG, RES  # noqa: E402  (shared rcParams)
from analyze_round9 import COLORS, NAMES, at, base_name, curve, fmt, load  # noqa: E402
from sweep_round10 import name as name10  # noqa: E402
from sweep_round12 import CONFIGS, POWERS, name  # noqa: E402

with open(os.path.join(RES, "round9_selection.json")) as f:
    SEL = json.load(f)
with open(os.path.join(RES, "round6_selection.json")) as f:
    SEL6 = json.load(f)


def ref_name(bs, m, dg):
    return name10(SEL, bs, m, "keep" if bs == 4096 else "lm", dg)  # batch 4096: 'safe_lm' == 'safe_keep' there


def baseline(opt, bs, which="lr_epoch10"):
    k = f"fm_bs{bs}_{opt}"
    lr = SEL[k][which] if k in SEL else SEL6[k]["lr"]
    return load(base_name(opt, lr), bs)


def main():
    lines = ["### Round 12: diagonal B0 = δ D (F-MNIST CNN, 10 epochs, 3 seeds)", "",
             "D = (√v̂ + 1e-8)^p normalized to mean 1, v the EMA of g² (β₂ = 0.999), refreshed every step; "
             "p = 0 is the reference (rounds 9-10). Training loss on the full training set after 10 epochs; "
             "optimizer FLOPs relative to the model's forward/backward FLOPs.", "",
             "| batch, setting | p | train loss | test acc % | accepted % | optimizer / model FLOPs | "
             "ratio to p = 0, per epoch |", "|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(20, 4.9))
    best = {}
    for ax, (bs, m, dg) in zip(axes, CONFIGS):
        setting = f"m = {m}" + (", double_grad" if dg else "")
        ref = load(ref_name(bs, m, dg), bs)
        Mref = np.array([curve(L)[1:11] for L in ref])
        for p, c, ls in ((0.0, "#8a8985", "-"), (POWERS[0], "#6250d6", "-"), (POWERS[1], "#2a78d6", "-")):
            logs = ref if p == 0 else load(name(SEL, bs, m, dg, p), bs)
            if not logs:
                continue
            M = np.array([curve(L)[1:11] for L in logs])
            r = np.exp(np.log(M).mean(0) - np.log(Mref).mean(0))
            acc = 100 * np.mean([d["acc"] for L in logs for d in L["iters"]])
            ovh = np.mean([L["epochs"][-1]["opt_gflops"] / L["epochs"][-1]["model_gflops"] for L in logs])
            tl = at(logs, 10)
            lines.append(f"| {bs}, {setting} | {p:g} | {fmt(*tl)} | {100 * at(logs, 10, 'test_acc')[0]:.2f} | "
                         f"{acc:.0f} | {100 * ovh:.1f}% | {'–' if p == 0 else f'{r.min():.2f}-{r.max():.2f}'} |")
            if p and (bs not in best or tl[0] < best[bs][1]):
                best[bs] = (p, tl[0], tl[1])
            ax.plot(np.arange(1, 11), np.exp(np.log(M).mean(0)), ls, marker="o", ms=3, lw=1.8, color=c,
                    label=f"ARCs-LSR1, {'B0 = δI (reference)' if p == 0 else f'diagonal B0, p = {p:g}'} (n={len(M)})")
        if bs in (1024, 4096):  # the diagonal B0 at its best floor (follow-up)
            from sweep_round12 import name_mu
            logs = load(name_mu(SEL, bs, m, dg, 1.0, 0.25), bs)
            if logs:
                M = np.array([curve(L)[1:11] for L in logs])
                ax.plot(np.arange(1, 11), np.exp(np.log(M).mean(0)), "-", marker="o", ms=3, lw=1.8, color="#1baf7a",
                        label=f"ARCs-LSR1, diagonal B0, p = 1, μ_min = 0.25 (n={len(M)})")
                tl = at(logs, 10)
                if tl[0] < best[bs][1]:
                    best[bs] = ("1, μ_min = 0.25", tl[0], tl[1])
        for opt, ls in (("adamw", "-."), ("sophia", (0, (3, 1, 1, 1))), ("shampoo", ":"), ("muon", "--")):
            b = baseline(opt, bs)
            if b:
                Mb = np.array([curve(L)[1:11] for L in b])
                ax.plot(np.arange(1, 11), np.exp(np.log(Mb).mean(0)), ls=ls, color=COLORS[opt], lw=1.3,
                        label=f"{NAMES[opt]} (tuned)")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=(2, 3, 4, 5, 6)))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.tick_params(axis="y", which="minor", labelsize=7)
        ax.set_xlabel("epoch")
        ax.set_title(f"batch {bs} ({setting})")
    axes[0].set_ylabel("training loss (geometric mean over seeds, log)")
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("F-MNIST CNN: ARCs-LSR1 with a diagonal initial matrix B0 = δD vs. the tuned baselines (per epoch)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(os.path.join(FIG, "fig17_diag_b0.png"), dpi=150)
    plt.close(fig)

    from sweep_round12 import name_mu
    mus = (0.0625, 0.25, 1.0, 4.0)
    lines += ["", "μ_min with and without the diagonal B0 (the round-9 value is μ_min = 1):", "",
              "| batch, B0 | " + " | ".join(f"μ_min = {mu:g}" for mu in mus) + " |", "|---|" + "---|" * len(mus)]
    for bs, m, dg in CONFIGS:
        if bs not in (1024, 4096):
            continue
        for p in (0.0, 1.0):
            row = []
            for mu in mus:
                if mu == 1.0:
                    logs = load(ref_name(bs, m, dg), bs) if p == 0 else load(name(SEL, bs, m, dg, p), bs)
                else:
                    logs = load(name_mu(SEL, bs, m, dg, p, mu), bs)
                row.append(fmt(*at(logs, 10)) if logs else "–")
            lines.append(f"| {bs}, {'δI' if p == 0 else 'δD, p = 1'} | " + " | ".join(row) + " |")

    lines += ["", "Best diagonal B0 vs. the tuned baselines, training loss after 10 epochs (3 seeds; baselines at "
              "their epoch-10 learning rate). In parentheses: the baselines after 20 epochs, i.e. at the same compute "
              "as 10 ARCs-LSR1 epochs (not run at batch 256).", "",
              "| batch | ARCs-LSR1, diagonal B0 | AdamW | SophiaG | Shampoo | Muon |", "|---|---|---|---|---|---|"]
    for bs, m, dg in CONFIGS:
        if bs not in best:
            continue
        p, v, sd = best[bs]
        row = [f"{v:.3f} ± {sd:.3f} (p = {p if isinstance(p, str) else f'{p:g}'})"]
        for opt in ("adamw", "sophia", "shampoo", "muon"):
            b10 = baseline(opt, bs)
            v10 = at(b10, 10)[0] if b10 else float("nan")
            b20 = baseline(opt, bs, "lr_epoch20") if f"fm_bs{bs}_{opt}" in SEL else []
            v20 = f" ({at(b20, 20)[0]:.3f})" if b20 and len(b20[0]["epochs"]) > 20 else ""
            row.append(f"{v10:.3f}{v20}")
        lines.append(f"| {bs} | " + " | ".join(row) + " |")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(RES, "summary_round12.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
