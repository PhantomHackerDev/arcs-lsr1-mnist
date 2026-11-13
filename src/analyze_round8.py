"""Round 8: does the L-SR1 curvature help when the gradient is less noisy?

Full-network ARCs-LSR1 (same-batch y, m=10) with and without curvature pairs
(no pairs: B = I, i.e. a cubic-regularized gradient step), over the floor mu_min,
on F-MNIST (CNN, batch 64 / 256 / 1024) and MNIST (deep MLP, batch 256).

  python src/analyze_round8.py -> results/summary_round8.md, figures/fig12_curvature_value.png
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
MUS = [0, 1, 4, 16, 64, 256]
# mu_min chosen by the seed-0 final training loss (the same rule as the baselines' lr), per column
SELECTED = {("fm", 64): (64, 16), ("fm", 256): (4, 4), ("fm", 1024): (4, 1), ("deep", 256): (4, 1)}  # 0 = no floor (the default mu_min = 1e-8)
SETTINGS = [  # (title, prefix used for the round-8 names, batch, name of the default run with pairs, sel key)
    ("F-MNIST CNN, batch 64", "fm", 64, "fm_full_same", "fm_bs64"),
    ("F-MNIST CNN, batch 256", "fm", 256, "fm_full_same", "fm_bs256"),
    ("F-MNIST CNN, batch 1024", "fm", 1024, None, None),
    ("MNIST deep MLP, batch 256", "deep", 256, "deep_full_same", "deep_bs256"),
]


def stem(prefix, bs, pairs, mu):
    """candidate run-name stems (without _bs/_s) for one cell; batch 64 reuses the round-7 runs"""
    if mu == 0 and pairs:
        return [{"fm": "fm_full_same", "deep": "deep_full_same"}[prefix]]
    out = [f"r8_{prefix}_{'pairs' if pairs else 'nopairs'}_mumin{mu}"]
    if bs == 64 and prefix == "fm" and mu:
        out.insert(0, f"r7_{'' if pairs else 'nopairs_'}mumin_{mu}")
    return out


def load(st, bs):
    for s_ in ([st] if isinstance(st, str) else st):
        out = []
        for f in sorted(glob.glob(os.path.join(RES, f"{s_}_bs{bs}_s[0-9].json"))):
            with open(f) as fh:
                out.append(json.load(fh))
        if out:
            return out
    return []


def final(logs, key):
    v = np.array([L["epochs"][-1][key] for L in logs], dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan")), len(v)


def cell(logs):
    if not logs:
        return "–"
    tl, tsd, n = final(logs, "train_loss")
    acc, _, _ = final(logs, "test_acc")
    p = 4 if tl < 0.05 else 3
    sd = f" ± {tsd:.{p}f}" if n > 1 else ""
    return f"{tl:.{p}f}{sd} ({100 * acc:.2f}%{', n=' + str(n) if n > 1 else ''})"


def main():
    lines = ["### Round 8: value of the L-SR1 curvature vs. the floor μ_min (10 epochs)", "",
             "Each cell: final training loss on the full training set (test accuracy). Seed 0 unless n is shown. "
             "'no pairs' stores no curvature pairs, so B = I and the step is a cubic-regularized gradient step "
             "(it still pays for the second gradient, which only its ratio test would need as a forward pass).", ""]
    fig, axs = plt.subplots(2, len(SETTINGS), figsize=(18, 8.6))
    axes = axs[0]
    for col, (ax, (title, prefix, bs, _, selk)) in enumerate(zip(axes, SETTINGS)):
        lines += [f"**{title}**", "", "| μ_min | with L-SR1 pairs | no pairs (B = I) |", "|---|---|---|"]
        curves = {True: [], False: []}
        for mu in MUS:
            row = [cell(load(stem(prefix, bs, p, mu), bs)) for p in (True, False)]
            lines.append(f"| {'none (default 1e-8)' if mu == 0 else mu} | {row[0]} | {row[1]} |")
            for p in (True, False):
                logs = load(stem(prefix, bs, p, mu), bs)
                if logs:
                    curves[p].append((mu or 0.25, final(logs, "train_loss")[0]))
        ref = []
        if selk:
            for o, n in (("muon", "Muon"), ("adamw", "AdamW")):
                logs = load(f"{prefix}_{o}_lr{SEL[f'{selk}_{o}']['lr']:g}", bs)
                if logs:
                    ref.append((n, final(logs, "train_loss")[0]))
            logs = load(f"{prefix}_adam", bs)
            if logs:
                ref.append(("Adam (paper)", final(logs, "train_loss")[0]))
        if ref:
            lines += ["", "Baselines, 3 seeds: " + ", ".join(f"{n} {v:.4g}" for n, v in ref)]
        lines.append("")
        # selected mu_min with vs. without pairs: final loss is noisy late in training
        # (MNIST), so also the geometric mean over epochs 1-10 and the per-epoch ratio
        mp, mn = SELECTED[(prefix, bs)]
        A = np.array([[e["train_loss"] for e in L["epochs"][1:]] for L in load(stem(prefix, bs, True, mp), bs)])
        B = np.array([[e["train_loss"] for e in L["epochs"][1:]] for L in load(stem(prefix, bs, False, mn), bs)])
        ga, gb = np.exp(np.log(A).mean(1)), np.exp(np.log(B).mean(1))
        r = np.exp(np.log(A).mean(0) - np.log(B).mean(0))
        lines += [f"Selected μ_min, pairs (μ_min = {mp}) vs. no pairs (μ_min = {mn}), {len(A)} / {len(B)} seeds: "
                  f"geometric mean of the training loss over epochs 1-10 {ga.mean():.4g} ± {ga.std(ddof=1):.2g} vs. "
                  f"{gb.mean():.4g} ± {gb.std(ddof=1):.2g}; ratio (pairs / no pairs) per epoch {r.min():.2f}-{r.max():.2f}.", ""]
        ax2 = axs[1, col]
        ep = np.arange(1, A.shape[1] + 1)
        for M, c, lab in ((A, "#2a78d6", f"L-SR1 pairs, μ_min = {mp}"), (B, "#eb6834", f"no pairs, μ_min = {mn}")):
            ax2.plot(ep, np.exp(np.log(M).mean(0)), "-o", color=c, ms=3, lw=1.8, label=f"{lab} (n={len(M)})")
            ax2.fill_between(ep, M.min(0), M.max(0), color=c, alpha=0.12, lw=0)
        if selk:
            for o, n_, ls in (("muon", "Muon", "--"), ("adam", "Adam (paper)", ":")):
                st = f"{prefix}_{o}_lr{SEL[f'{selk}_{o}']['lr']:g}" if o == "muon" else f"{prefix}_adam"
                M = np.array([[e["train_loss"] for e in L["epochs"][1:]] for L in load(st, bs)])
                if len(M):
                    ax2.plot(ep, np.exp(np.log(M).mean(0)), ls, color="#3d3c39", lw=1.3, label=n_)
        ax2.set_yscale("log")
        ax2.set_xlabel("epoch")
        ax2.legend(fontsize=8)

        for p, c, lab in ((True, "#2a78d6", "ARCs-LSR1 (L-SR1 pairs)"), (False, "#eb6834", "no pairs (B = I)")):
            if curves[p]:
                x, y = zip(*curves[p])
                ax.plot(x, y, "-o", color=c, ms=4, lw=1.8, label=lab)
        for (n, v), ls in zip(ref, ("--", ":", "-.")):
            ax.axhline(v, color="#3d3c39", ls=ls, lw=1.1, label=n)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([0.25, 1, 4, 16, 64, 256])
        ax.set_xticklabels(["none", "1", "4", "16", "64", "256"])
        ax.set_xlabel("floor μ_min")
        ax.set_title(title)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc="best", framealpha=0.9)
    for col, (_, prefix, _, _, _) in enumerate(SETTINGS):
        for a_ in axs[:, col]:
            a_.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            a_.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g") if prefix == "fm"
                                         else matplotlib.ticker.NullFormatter())
            a_.tick_params(axis="y", which="minor", labelsize=7)
    axes[0].set_ylabel("final training loss (log)")
    axs[1, 0].set_ylabel("training loss (geometric mean over seeds;\nband = min-max)")
    fig.suptitle("Top: final training loss vs. the floor μ_min (seed 0 unless averaged), with and without L-SR1 "
                 "curvature pairs. Bottom: training loss per epoch at the selected μ_min",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, "fig12_curvature_value.png"), dpi=150)
    plt.close(fig)
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(RES, "summary_round8.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
