"""Round 7: per-batch overfitting on F-MNIST (CNN, batch 64, full-network ARCs-LSR1, m=10).

  (R) ratio test on a larger batch (the current batch + random other samples)
  (Y) curvature pairs and ratio test on that larger batch
  (H) ARCs hyperparameters, one at a time from the defaults (eta1=0.1, eta2=0.75,
      mu0=1, mu_min=1e-8, gamma rule 'safe')
Selection is by final TRAINING loss, as for the baselines' learning rates.

  python src/analyze_round7.py -> results/summary_round7.md, figures/fig11_round7.png
"""
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from analyze import FIG, RES  # noqa: E402  (shared rcParams)

BS = 64
with open(os.path.join(RES, "round6_selection.json")) as f:
    SEL = json.load(f)
ETAS = ["eta0.1-0.75", "eta0.01-0.25", "eta0-0.1"]
GROUPS = [
    ("R: ratio test on a larger batch (gradient and pairs on the current 64)",
     [f"r7_rho{nb}_{e}" for nb in (256, 1024) for e in ETAS]),
    ("Y: curvature pairs and ratio test on the larger batch",
     [f"r7_yrho{nb}_{e}" for nb in (256, 1024) for e in ETAS[1:]]),
    ("H: hyperparameters (everything on the current batch)",
     ["fm_full_same", "r7_eta1_0", "r7_eta1_0.01", "r7_eta1_0.25", "r7_eta1_0.5", "r7_eta2_0.25", "r7_eta2_0.5",
      "r7_eta2_0.9", "r7_eta0.01-0.25", "r7_eta0-0.1", "r7_mu0_0.01", "r7_mu0_100", "r7_mumin_1", "r7_mumin_4",
      "r7_mumin_16", "r7_mumin_64", "r7_mumin_256", "r7_eta1_0.7", "r7_bb"]),
    ("A: ablation, same μ_min without curvature pairs (B = I)",
     ["r7_nopairs_mumin_4", "r7_nopairs_mumin_16", "r7_nopairs_mumin_64"]),
    ("C: combinations with μ_min", ["r7_mumin16_eta1_0.5", "r7_mumin16_rho256_eta0.01-0.25",
                                    "r7_mumin16_yrho256_eta0.01-0.25"]),
]
BASELINES = [("fm_sgd", "SGD+momentum (paper: lr 0.1)"), ("fm_adam", "Adam (paper: lr 1e-3)")] + [
    (f"fm_{o}_lr{SEL[f'fm_bs{BS}_{o}']['lr']:g}", f"{n} (tuned lr {SEL[f'fm_bs{BS}_{o}']['lr']:.2g})")
    for o, n in (("adamw", "AdamW"), ("muon", "Muon"), ("shampoo", "Shampoo"), ("sophia", "SophiaG"))]


def label(stem):
    if stem == "fm_full_same":
        return "defaults (round 5)"
    a = re.sub(r"eta([\d.]+)-([\d.]+)", r"(η₁, η₂) = (\1, \2)", stem[3:])
    a = re.sub(r"^(y?)rho(\d+)_", lambda m: ("pairs + ratio" if m.group(1) else "ratio") + f" on {m.group(2)} samples, ", a)
    a = a.replace("eta1_", "η₁ = ").replace("eta2_", "η₂ = ").replace("mu0_", "μ₀ = ").replace("mumin_", "μ_min = ")
    a = re.sub(r"^mumin(\d+)_", r"μ_min = \1, ", a).replace("nopairs_", "no pairs (B = I), ")
    a = re.sub(r", (y?)rho(\d+)_", lambda m: ", " + ("pairs + ratio" if m.group(1) else "ratio") + f" on {m.group(2)} samples, ", a)
    return a.replace("bb", "γ rule: BB")


def load(stem, seeds=None):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, f"{stem}_bs{BS}_s[0-9].json"))):
        if seeds is None or int(f[-6]) in seeds:
            with open(f) as fh:
                out.append(json.load(fh))
    return out


def compute(L):
    """model compute per epoch, in units of one full-training-set gradient evaluation"""
    out = []
    F = L.get("model_flops_per_sample") or {}
    for e in L["epochs"][1:]:
        if "grad_epochs" in e:
            out.append(e["grad_epochs"])
        elif F:
            fwd = e.get("n_fwd", 0) * F["forward"] / F["forward_backward"]
            out.append((e["n_grad"] + fwd) * BS / 60000)
        else:
            out.append(e["n_grad"] * BS / 60000)
    return np.array(out)


def stats(stem, seeds=None):
    logs = load(stem, seeds)
    if not logs:
        return None
    tl = np.array([[e["train_loss"] for e in L["epochs"]] for L in logs])
    acc = np.array([[e["test_acc"] for e in L["epochs"]] for L in logs]) * 100
    r = {"n": len(logs), "tl": tl, "acc": acc, "comp": np.array([compute(L) for L in logs]).mean(0)}
    its = [L["iters"] for L in logs if L["iters"] and "acc" in L["iters"][0]]
    if its:
        last = [it[-len(it) // 10:] for it in its]
        r["accept"] = 100 * np.mean([d["acc"] for it in its for d in it])
        r["mu_last"] = float(np.median([d["mu"] for it in last for d in it]))
        r["s_last"] = float(np.median([d["s_norm"] for it in last for d in it if d["acc"]] or [0.0]))
    return r


def fmt(v, sd=None, p=4):
    return f"{v:.{p}f}" + (f" ± {sd:.{p}f}" if sd is not None else "")


def row(stem, name, r):
    tl, acc = r["tl"][:, -1], r["acc"][:, -1]
    sd = (lambda a: a.std(ddof=1)) if r["n"] > 1 else (lambda a: None)
    extra = (f"{r['accept']:.0f} | {r['mu_last']:.3g} | {r['s_last']:.3g}" if "accept" in r else "– | – | –")
    return (f"| {name} | {r['n']} | {fmt(tl.mean(), sd(tl))} | {fmt(acc.mean(), sd(acc), 2)} | "
            f"{r['tl'][:, 1].mean():.3f} / {r['tl'][:, 3].mean():.3f} | {extra} | {r['comp'][-1] / 10:.1f} |")


HEAD = ["| run | seeds | final train loss | test acc % | train loss ep 1 / 3 | accepted % | median μ, last 10% |"
        " median ‖s‖ (accepted), last 10% | compute / epoch (full-gradient units) |",
        "|---|---|---|---|---|---|---|---|---|"]


def main():
    lines = ["### Round 7: F-MNIST CNN, batch 64, full-network ARCs-LSR1 (m=10, same-batch y), 10 epochs", "",
             "Seed 0 for screening; the selected configurations also have seeds 1-2 (seeds column). "
             "Compute counts every forward/backward sample, so passes on the larger batch are charged at their "
             "size; 1.0 = one full-training-set gradient.", ""]
    best = {}
    for title, stems in GROUPS:
        lines += [f"**{title}**", ""] + HEAD
        for stem in stems:
            r0 = stats(stem, seeds={0})
            if r0 is None:
                continue
            r = stats(stem)
            lines.append(row(stem, label(stem), r))
            if stem != "fm_full_same":
                key = title[0]
                if key not in best or r0["tl"][0, -1] < best[key][1]:
                    best[key] = (stem, r0["tl"][0, -1])
        lines.append("")
    lines += ["**Baselines (round 6)**", ""] + HEAD
    for stem, name in BASELINES:
        r = stats(stem)
        if r:
            lines.append(row(stem, name, r))
    lines += ["", "Best by seed-0 training loss: " + ", ".join(f"{k}: {v[0]} ({v[1]:.4f})" for k, v in best.items()), ""]

    # figure: training loss vs epoch and vs compute
    series = [("fm_full_same", "ARCs-LSR1, defaults", "#2a78d6", "-")]
    cols = {"R": "#eb6834", "Y": "#6250d6", "H": "#1baf7a", "C": "#e87ba4", "A": "#8a8985"}
    for k, (stem, _) in best.items():
        series.append((stem, f"ARCs-LSR1, best {k}: {label(stem)}", cols[k], "-"))
    series += [(BASELINES[1][0], BASELINES[1][1], "#3d3c39", "--"), (BASELINES[0][0], BASELINES[0][1], "#8a8985", "--"),
               (BASELINES[3][0], BASELINES[3][1], "#eda100", "--")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for stem, name, c, ls in series:
        r = stats(stem)
        if r is None:
            continue
        ep = np.arange(r["tl"].shape[1])
        axes[0].plot(ep[1:], r["tl"].mean(0)[1:], ls, color=c, marker="o", ms=3, lw=1.8, label=f"{name} (n={r['n']})")
        axes[1].plot(r["comp"], r["tl"].mean(0)[1:], ls, color=c, marker="o", ms=3, lw=1.8)
        axes[2].plot(ep[1:], r["acc"].mean(0)[1:], ls, color=c, marker="o", ms=3, lw=1.8)
    for ax in axes[:2]:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel("training loss (full training set, log)")
    axes[1].set_xscale("log")
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axes[0].set_xlabel("epoch")
    axes[1].set_xlabel("model compute (full-training-set gradient evaluations, log)")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("test accuracy (%)")
    axes[0].set_title("Training loss per epoch")
    axes[1].set_title("Training loss per unit of compute")
    axes[2].set_title("Test accuracy")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("F-MNIST CNN, batch 64: fixes for per-batch overfitting vs. baselines (mean over seeds)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.savefig(os.path.join(FIG, "fig11_round7.png"), dpi=150)
    plt.close(fig)
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(RES, "summary_round7.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
