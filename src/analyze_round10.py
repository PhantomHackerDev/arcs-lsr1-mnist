"""Round 10: (a) LM-damped delta rule 'safe_lm', (b) gradient over two batches ('double_grad').

  python src/analyze_round10.py -> results/summary_round10.md, figures/fig15_round10.png
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from analyze import FIG, RES  # noqa: E402  (shared rcParams)
from analyze_round9 import at, base_name, curve, fmt, lam_neg, load  # noqa: E402
from sweep_round9m import name as name9  # noqa: E402
from sweep_round10 import name, setting  # noqa: E402

with open(os.path.join(RES, "round9_selection.json")) as f:
    SEL = json.load(f)
with open(os.path.join(RES, "round6_selection.json")) as f:
    SEL6 = json.load(f)  # baselines at batch 256 (10 epochs)


def col(logs, key, f=np.median):
    v = [np.ravel(d[key])[0] for L in logs for d in L["iters"] if key in d]
    v = np.array(v, dtype=float)
    return f(v[np.isfinite(v)]) if len(v) else float("nan")


def row(lab, logs, extra=""):
    tl, ta = at(logs, 10), at(logs, 10, "test_acc")
    acc = 100 * col(logs, "acc", np.mean)
    comp = np.mean([L["epochs"][-1]["grad_epochs"] for L in logs])
    return f"| {lab} | {len(logs)} | {fmt(*tl)} | {100 * ta[0]:.2f} | {acc:.0f} | {lam_neg(logs):.1f} | {comp:.1f} |{extra}"


def main():
    lines = ["### Round 10: two ideas from Goldfarb, Ren & Bahamou (2020), F-MNIST CNN", "",
             "ARCs-LSR1, full network, same-batch y, 10 epochs; train loss on the full training set after 10 epochs "
             "(test accuracy); compute in full-training-set gradient evaluations.", "",
             "#### (a) LM damping in the δ rule ('safe_lm')", "",
             "| batch, m, δ rule | seeds | train loss | test acc % | accepted % | λ̂_min ≤ 0 (% of steps) | compute | "
             "median LM shift when applied |", "|---|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))
    ax = axes[0]
    e2, mu = setting(SEL, 2048)
    for rule, c, mk in (("safe", "#eb6834", "s"), ("keep", "#8a8985", "^"), ("lm", "#2a78d6", "o")):
        pts = []
        for m in (10, 20, 50, 100, 200):
            if rule == "safe":
                logs = load(SEL["fm_bs2048_arcs_pairs"]["name"], 2048) if m == 10 else (load(name9(SEL, 2048, m, False), 2048) if m < 200 else [])
            elif rule == "keep":
                logs = load(name(SEL, 2048, m, "keep") if m == 200 else name9(SEL, 2048, m, True), 2048)
            else:
                logs = load(name(SEL, 2048, m, "lm"), 2048) if m != 10 else []
            if not logs:
                continue
            lm_med = col([L for L in logs], "lm", lambda v: np.median(v[v > 0]) if (v > 0).any() else 0.0)
            lines.append(row(f"2048, m = {m}, {'safe' if rule == 'safe' else 'safe_' + rule}", logs,
                             f" {lm_med:.3g} |" if rule == "lm" else " – |"))
            v = np.array([curve(L)[10] for L in logs])
            pts.append((m, v.mean(), v.min(), v.max()))
        if pts:
            m_, y, lo_, hi_ = map(np.array, zip(*pts))
            ax.errorbar(m_, y, yerr=[y - lo_, hi_ - y], color=c, marker=mk, ms=5, lw=1.8, capsize=3,
                        label={"safe": "safe (paper)", "keep": "safe_keep", "lm": "safe_lm (LM damping)"}[rule])
    for bs in (1024, 4096):
        for rule in ("keep", "lm"):
            logs = load(name(SEL, bs, 200, rule), bs)
            if logs:
                lm_med = col(logs, "lm", lambda v: np.median(v[v > 0]) if (v > 0).any() else 0.0)
                lines.append(row(f"{bs}, m = 200, safe_{rule}", logs, f" {lm_med:.3g} |" if rule == "lm" else " – |"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([10, 20, 50, 100, 200])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=(2, 3, 4, 5, 6)))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.tick_params(axis="y", which="minor", labelsize=7)
    ax.set_xlabel("L-SR1 memory m")
    ax.set_ylabel("training loss after 10 epochs (log)")
    ax.set_title("(a) batch 2048: δ rule vs. memory\n(3 seeds; safe at m = 20, 50 and safe_keep at m = 200: seed 0)",
                 fontsize=10)
    ax.legend(fontsize=8)

    lines += ["", "#### (b) gradient over two batches at no extra cost ('double_grad')", "",
              "m = 50, δ rule safe_keep. 'double_grad': model gradient = mean of the current and the previous batch's "
              "gradient at the current point. '+ ratio': the ratio is also measured on the two-batch mean loss "
              "(one extra forward pass per step).", "",
              "| batch, variant | seeds | train loss | test acc % | accepted % | λ̂_min ≤ 0 (% of steps) | compute | "
              "geometric mean over epochs 1-10 | ratio to reference, per epoch |", "|---|---|---|---|---|---|---|---|---|"]
    for ax, bs in zip(axes[1:], (256, 1024, 2048, 4096)):
        refs = {}
        for rule, dg, lab, c, ls in (("keep", "", "reference (safe_keep)", "#8a8985", "-"),
                                     ("keep", "_dg", "double_grad (safe_keep)", "#2a78d6", "--"),
                                     ("keep", "_dgr", "double_grad + ratio (safe_keep)", "#1baf7a", "--"),
                                     ("lm", "", "reference (safe_lm)", "#3d3c39", ":"),
                                     ("lm", "_dg", "double_grad (safe_lm)", "#2a78d6", "-")):
            logs = load(name(SEL, bs, 50, rule, dg), bs)
            if not logs and rule == "lm" and dg == "":
                # 'safe_lm' == 'safe_keep' when lambda_hat_min > 0 in every step
                keep = load(name(SEL, bs, 50, "keep", ""), bs)
                if keep and lam_neg(keep) == 0:
                    logs, lab = keep, "reference (safe_lm = safe_keep here)"
            if not logs:
                continue
            if dg == "":
                refs[rule] = np.array([curve(L)[1:11] for L in logs])
            Mref = refs.get(rule)
            M = np.array([curve(L)[1:11] for L in logs])
            gm = np.exp(np.log(M).mean(1))
            r = (np.exp(np.log(M).mean(0) - np.log(Mref).mean(0)) if Mref is not None and len(Mref) else None)
            rs = f"{r.min():.2f}-{r.max():.2f}" if (r is not None and dg) else "–"
            lines.append(row(f"{bs}, {lab}", logs, f" {gm.mean():.4g} | {rs} |"))
            ax.plot(np.arange(1, 11), np.exp(np.log(M).mean(0)), ls, marker="o", color=c, ms=3, lw=1.8,
                    label=f"{lab} (n={len(M)})")
        for opt, ls in (("adamw", "-."), ("muon", "--")):
            k = f"fm_bs{bs}_{opt}"
            lr = SEL[k]["lr_epoch10"] if k in SEL else (SEL6[k]["lr"] if k in SEL6 else None)
            if lr is not None:
                b = load(base_name(opt, lr), bs)
                if b:
                    ax.plot(np.arange(1, 11), np.exp(np.log(np.array([curve(L)[1:11] for L in b])).mean(0)), ls,
                            color="#3d3c39", lw=1.1, label=f"{'AdamW' if opt == 'adamw' else 'Muon'} (tuned)")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=(2, 3, 5)))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.tick_params(axis="y", which="minor", labelsize=7)
        ax.set_xlabel("epoch")
        ax.set_title(f"(b) batch {bs}: training loss per epoch")
        ax.legend(fontsize=7)
    fig.suptitle("Round 10: (a) LM damping in the δ rule, (b) model gradient over two batches (m = 50); "
                 "mean over seeds (geometric for the curves)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG, "fig15_round10.png"), dpi=150)
    plt.close(fig)
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(RES, "summary_round10.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
