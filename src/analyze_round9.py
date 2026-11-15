"""Round 9: larger batches (F-MNIST CNN, batch 1024 / 2048 / 4096).

  python src/analyze_round9.py -> results/summary_round9.md, figures/fig13_large_batch.png
Needs results/round9_selection.json (written by src/sweep_round9.py).
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
from sweep_round9 import BSS, ETA2, LRS, arcs_name, base_name  # noqa: E402

MUS = (1e-8, 1, 4, 16, 64, 256, 1024, 4096)
NAMES = {"sgd": "SGD+momentum", "adamw": "AdamW", "muon": "Muon", "shampoo": "Shampoo", "sophia": "SophiaG",
         "adam": "Adam (paper, lr 1e-3)"}
COLORS = {"pairs": "#2a78d6", "nopairs": "#eb6834", "muon": "#3d3c39", "shampoo": "#1baf7a", "adamw": "#6250d6",
          "sophia": "#eda100", "sgd": "#8a8985", "adam": "#e87ba4"}


def load(name, bs, seeds=(0, 1, 2)):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, f"{name}_bs{bs}_s[0-9].json"))):
        if int(f[-6]) in seeds:
            with open(f) as fh:
                out.append(json.load(fh))
    return out


def curve(L, key="train_loss"):
    return np.array([e[key] for e in L["epochs"]], dtype=float)


def compute(L):
    """cumulative model compute in full-training-set gradient evaluations, per epoch (epoch 0 = 0)"""
    return np.array([e.get("grad_epochs", e["epoch"]) if e["epoch"] else 0.0 for e in L["epochs"]])


def at(logs, ep, key="train_loss"):
    v = np.array([curve(L, key)[ep] for L in logs])
    v = np.where(np.isfinite(v), v, np.inf)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else None), len(v)


def fmt(m, sd, n, p=None):
    p = p or (4 if m < 0.05 else 3)
    return f"{m:.{p}f}" + (f" ± {sd:.{p}f}" if sd is not None else "") + (f" (n={n})" if n > 1 else "")


def arc_stats(logs, mu_min):
    its = [d for L in logs for d in L["iters"] if "acc" in d]
    if not its:
        return "– | –"
    acc = 100 * np.mean([d["acc"] for d in its])
    above = 100 * np.mean([d["mu"] > 1.001 * mu_min for d in its])
    return f"{acc:.0f} | {above:.0f}"


def lam_neg(logs):
    """share of steps (after the first) where lambda_hat_min <= 0, i.e. where 'safe' falls back to delta = 1"""
    v = [np.ravel(d["lam_hat_min"])[0] for L in logs for d in L["iters"][1:] if d.get("lam_hat_min") is not None]
    v = np.array(v, dtype=float)
    return 100 * float(np.mean(v[np.isfinite(v)] <= 0)) if len(v) else float("nan")


def memory_section(SEL):
    """L-SR1 memory m at the selected (eta2, mu_min); see src/sweep_round9m.py"""
    from sweep_round9m import MS_KEEP, MS_SAFE, name
    lines = ["#### L-SR1 memory m (selected η₂ and μ_min per batch size)", "",
             "'safe' is the paper's γ rule (γ = λ̂_min / 2, or γ = 1 when λ̂_min ≤ 0); 'safe_keep' keeps the previous "
             "γ when λ̂_min ≤ 0. Training loss after 10 epochs (≈ 20 full-gradient evaluations); the optimizer's own "
             "arithmetic is given relative to the model's forward/backward FLOPs.", "",
             "| batch | m | γ rule | seeds | train loss | test acc % | accepted % | median γ | λ̂_min ≤ 0 (% of steps) "
             "| optimizer / model FLOPs |", "|---|---|---|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, len(BSS), figsize=(17, 4.8))
    for ax, bs in zip(axes, BSS):
        sel = SEL[f"fm_bs{bs}_arcs_pairs"]
        rows = [(m, False, load(name(SEL, bs, m, False), bs)) for m in MS_SAFE]
        rows.insert(1, (10, False, load(sel["name"], bs)))
        rows += [(m, True, load(name(SEL, bs, m, True), bs)) for m in MS_KEEP]
        pts = {False: [], True: []}
        for m, keep, logs in rows:
            if not logs:
                continue
            tl, ta = at(logs, 10), at(logs, 10, "test_acc")
            its = [d for L in logs for d in L["iters"]]
            acc = 100 * np.mean([d["acc"] for d in its])
            gam = np.median([np.ravel(d["delta"])[0] for d in its])
            ovh = np.mean([L["epochs"][-1]["opt_gflops"] / L["epochs"][-1]["model_gflops"] for L in logs])
            lines.append(f"| {bs} | {m} | {'safe_keep' if keep else 'safe'} | {len(logs)} | {fmt(*tl)} | "
                         f"{100 * ta[0]:.2f} | {acc:.0f} | {gam:.3g} | {lam_neg(logs):.1f} | {100 * ovh:.2f}% |")
            v = np.array([curve(L)[10] for L in logs])
            pts[keep].append((m, v.mean(), v.min(), v.max(), len(v)))
        for keep, c, mk, lab in ((True, "#2a78d6", "o", "ARCs-LSR1, γ rule safe_keep"),
                                 (False, "#eb6834", "s", "ARCs-LSR1, γ rule safe (paper)")):
            if pts[keep]:
                m_, y, lo_, hi_, n_ = map(np.array, zip(*sorted(pts[keep])))
                ax.errorbar(m_, y, yerr=[y - lo_, hi_ - y], color=c, marker=mk, ms=5, lw=1.8, capsize=3,
                            ls="-" if keep else "--", mfc=c if keep else "white", label=lab)
        for opt, ls in (("muon", "--"), ("shampoo", ":"), ("adamw", "-.")):
            logs = load(base_name(opt, SEL[f"fm_bs{bs}_{opt}"]["lr_epoch10"]), bs)
            if logs:
                ax.axhline(at(logs, 10)[0], color=COLORS[opt], ls=ls, lw=1.2, label=f"{NAMES[opt]} (tuned, epoch 10)")
        logs = load(base_name("sgd", SEL[f"fm_bs{bs}_sgd"]["lr_epoch10"]), bs)
        if logs:
            ax.axhline(at(logs, 10)[0], color=COLORS["sgd"], ls=(0, (1, 3)), lw=1.2, label="SGD+momentum (tuned)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([5, 10, 20, 50, 100])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=(2, 3, 4, 5, 6)))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.tick_params(axis="y", which="minor", labelsize=7)
        ax.set_xlabel("L-SR1 memory m (log)")
        ax.set_title(f"batch {bs}")
    axes[0].set_ylabel("training loss after 10 epochs (log)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(("F-MNIST CNN: ARCs-LSR1 training loss vs. the L-SR1 memory m (safe_keep: 3 seeds, bars = min-max; "
                 "safe: seed 0 except m = 10)"),
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(os.path.join(FIG, "fig14_memory.png"), dpi=150)
    plt.close(fig)
    return lines + [""]


def main():
    with open(os.path.join(RES, "round9_selection.json")) as f:
        SEL = json.load(f)
    lines = ["### Round 9: larger batches, F-MNIST CNN", "",
             "ARCs-LSR1: full network, same-batch y, m = 10, η₁ = 0.1, 10 epochs (each step costs two gradients, "
             "so 10 epochs ≈ 20 full-gradient evaluations). Baselines: 20 epochs, so they can be compared both at "
             "the same epoch (10) and at the same compute (20 full-gradient evaluations). All selection is by the "
             "seed-0 final training loss; the selected configurations have seeds 0-2.", ""]
    fig, axs = plt.subplots(2, len(BSS), figsize=(18, 9.2))
    for col, bs in enumerate(BSS):
        lines += [f"#### Batch {bs} ({60000 // bs} steps per epoch)", "",
                  "ARCs-LSR1 grid, seed 0, final training loss after 10 epochs:", "",
                  "| μ_min | " + " | ".join(f"η₂ = {e:g}, pairs | η₂ = {e:g}, B = I" for e in ETA2) + " |",
                  "|---|" + "---|" * (2 * len(ETA2))]
        for mu in MUS:
            row = []
            for e2 in ETA2:
                for pairs in (True, False):
                    L = load(arcs_name(pairs, e2, mu), bs, seeds=(0,))
                    row.append(f"{curve(L[0])[-1]:.4f}" if L else "–")
            if any(r != "–" for r in row):
                lines.append(f"| {'none' if mu < 1e-3 else f'{mu:g}'} | " + " | ".join(row) + " |")
        lines += ["", "Selected configurations (mean ± sd over seeds; compute in full-gradient evaluations):", "",
                  "| method | setting | train loss, epoch 10 | test acc %, epoch 10 | train loss at 20 grads "
                  "| test acc % at 20 grads | accepted % | μ above floor % |",
                  "|---|---|---|---|---|---|---|---|"]
        rows = []  # (label, name, setting, ep10 logs, ep20 logs-or-same, color, ls)
        for p in ("pairs", "nopairs"):
            s = SEL[f"fm_bs{bs}_arcs_{p}"]
            logs = load(s["name"], bs)
            lab = "ARCs-LSR1" if p == "pairs" else "ARCs, no pairs (B = I)"
            mu_s = "none" if s["mu_min"] < 1e-3 else f"{s['mu_min']:g}"
            setting = f"η₂ = {s['eta2']:g}, μ_min = {mu_s}"
            tl10, ta10 = at(logs, 10), at(logs, 10, "test_acc")
            lines.append(f"| {lab} | {setting} | {fmt(*tl10)} | {100 * ta10[0]:.2f} | {fmt(*tl10)} | "
                         f"{100 * ta10[0]:.2f} | {arc_stats(logs, s['mu_min'])} |")
            rows.append((lab, logs, logs, COLORS[p], "-"))
        for opt in list(LRS) + ["adam"]:
            if opt == "adam":
                l10 = l20 = load(base_name("adam", 1e-3), bs)
                setting = "lr 1e-3"
            else:
                s = SEL[f"fm_bs{bs}_{opt}"]
                l10, l20 = load(base_name(opt, s["lr_epoch10"]), bs), load(base_name(opt, s["lr_epoch20"]), bs)
                setting = (f"lr {s['lr_epoch10']:.3g}" + ("" if s["lr_epoch20"] == s["lr_epoch10"]
                                                          else f" / {s['lr_epoch20']:.3g}")
                           + (" (edge)" if s["at_edge"] else ""))
            if not l10 or not l20:
                continue
            tl10, ta10 = at(l10, 10), at(l10, 10, "test_acc")
            tl20, ta20 = at(l20, 20), at(l20, 20, "test_acc")
            lines.append(f"| {NAMES[opt]} | {setting} | {fmt(*tl10)} | {100 * ta10[0]:.2f} | {fmt(*tl20)} | "
                         f"{100 * ta20[0]:.2f} | – | – |")
            rows.append((NAMES[opt], l10, l20, COLORS[opt], "--"))
        lines += ["", "Baseline lr: chosen at epoch 10 / at epoch 20 when they differ; '(edge)' = best at the edge "
                  "of the sweep after the extensions.", ""]
        # figure: top = per epoch (first 10 epochs, lr chosen at epoch 10); bottom = per compute (lr at epoch 20)
        for lab, l10, l20, c, ls in rows:
            M = np.array([curve(L)[1:11] for L in l10])
            axs[0, col].plot(np.arange(1, 11), np.exp(np.log(M).mean(0)), ls, color=c, marker="o", ms=3, lw=1.8,
                             label=f"{lab} (n={len(M)})")
            n = min(len(curve(L)) for L in l20)
            M = np.array([curve(L)[1:n] for L in l20])
            X = np.array([compute(L)[1:n] for L in l20]).mean(0)
            axs[1, col].plot(X, np.exp(np.log(M).mean(0)), ls, color=c, marker="o", ms=2.5, lw=1.8)
        axs[0, col].set_title(f"batch {bs}: training loss per epoch")
        axs[1, col].set_title(f"batch {bs}: training loss per unit of compute")
        axs[0, col].set_xlabel("epoch")
        axs[1, col].set_xlabel("model compute (full-training-set gradient evaluations)")
        axs[1, col].axvline(20, color="#8a8985", lw=0.8, ls=":")
        for ax in axs[:, col]:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=(2, 3, 5)))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.tick_params(axis="y", which="minor", labelsize=7)
    axs[0, 0].set_ylabel("training loss (geometric mean over seeds, log)")
    axs[1, 0].set_ylabel("training loss (geometric mean over seeds, log)")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("F-MNIST CNN at larger batches: tuned ARCs-LSR1 (with and without curvature pairs) vs. tuned "
                 "baselines. Top: per epoch. Bottom: per compute (baselines run 20 epochs)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(os.path.join(FIG, "fig13_large_batch.png"), dpi=150)
    plt.close(fig)
    lines += memory_section(SEL)
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(RES, "summary_round9.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
