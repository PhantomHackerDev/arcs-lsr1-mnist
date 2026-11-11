"""Aggregate results/*.json into a summary table and figures.

  python src/analyze.py            -> results/summary.md, results/summary.csv, figures/*.png
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# ---- fixed entity -> color (reference categorical palette, fixed order) ----
C = {
    "safe_cross": "#2a78d6",  # 1 blue    ARCs-LSR1, paper loop (cross-batch y)
    "safe_same": "#eb6834",  # 2 orange  Direction 3: same-batch y
    "safe_cross_layer": "#1baf7a",  # 3 aqua   Direction 1
    "paper_cross": "#eda100",  # 4 yellow
    "bb_cross": "#e87ba4",  # 5 magenta
    "safe_same_layer": "#008300",  # 6 green
    "safe_cross_gb0.9": "#6250d6",  # 7 violet  Direction 2
    "bb_same": "#e34948",  # 8 red
    "paper_same": "#eda100",
    "safe_same_gb0.9": "#6250d6",
    "sgd": "#8a8985",
    "adam": "#3d3c39",
}
LABEL = {
    "sgd": "SGD+momentum (lr 0.1)",
    "adam": "Adam (lr 1e-3)",
    "paper_cross": "ARCs-LSR1, γ = λ̂_min (Alg. 1 literal)",
    "safe_cross": "ARCs-LSR1, γ = ½λ̂_min (safeguarded)",
    "bb_cross": "ARCs-LSR1, γ = yᵀy/sᵀy",
    "paper_same": "γ = λ̂_min + same-batch y",
    "safe_same": "ARCs-LSR1 + same-batch (s, y)  [Dir. 3]",
    "bb_same": "γ = yᵀy/sᵀy + same-batch y",
    "safe_cross_layer": "per-layer subproblems  [Dir. 1]",
    "safe_same_layer": "per-layer + same-batch y  [Dir. 1+3]",
    "safe_cross_gb0.9": "gradient momentum β=0.9  [Dir. 2]",
    "safe_same_gb0.9": "momentum β=0.9 + same-batch y  [Dir. 2+3]",
}
ORDER = ["sgd", "adam", "paper_cross", "safe_cross", "bb_cross", "paper_same", "safe_same", "bb_same",
         "safe_cross_layer", "safe_same_layer", "safe_cross_gb0.9", "safe_same_gb0.9"]
BS = [64, 256, 1024]

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "text.color": "#0b0b0b", "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


def load():
    runs = defaultdict(list)  # (config, bs) -> [log]
    for f in glob.glob(os.path.join(RES, "*.json")):
        m = re.match(r"(.+)_bs(\d+)_s(\d+)\.json$", os.path.basename(f))
        if not m:
            continue
        with open(f) as fh:
            runs[(m.group(1), int(m.group(2)))].append(json.load(fh))
    return runs


def run_metrics(L):
    ep = L["epochs"]
    out = {
        "final_acc": ep[-1]["test_acc"], "best_acc": max(e["test_acc"] for e in ep),
        "final_train_loss": ep[-1]["train_loss"], "time": ep[-1]["time"], "n_grad": ep[-1]["n_grad"],
        "epochs_run": ep[-1]["epoch"],
    }
    hit = [e["epoch"] for e in ep if e["test_acc"] >= 0.97]
    out["ep_to_97"] = hit[0] if hit else float("nan")
    it = L["iters"]
    if it and "mu" in it[0]:
        mu = np.array([d["mu"] for d in it])
        out["accept"] = float(np.mean([d["acc"] for d in it]))
        out["mu_median"] = float(np.median(mu))
        out["mu_max"] = float(mu.max())
        out["frac_mu_gt_1e6"] = float(np.mean(mu > 1e6))
        out["frac_indef"] = float(np.mean([sum(d["n_neg"]) > 0 for d in it]))
        cm = np.array([max(d["cond_minv"]) for d in it if min(d["k_pairs"]) > 0])
        cm = cm[np.isfinite(cm)] if cm.size else cm
        out["cond_minv_median"] = float(np.median(cm)) if cm.size else float("nan")
        out["frac_cond_gt_1e8"] = float(np.mean(cm > 1e8)) if cm.size else float("nan")
        eig = np.array([max(d["eig_max_abs"]) for d in it])
        out["eig_max_abs_max"] = float(eig.max())
        last = ep[-1]
        out["skip_rate"] = float(sum(last["n_skip"]) / max(1, sum(last["n_skip"]) + sum(last["n_add"])))
        pr = L["probes"]
        neg = [p for p in pr if "lamB_neg" in p]
        out["n_neg_probes"] = len(neg)
        out["neg_real_frac"] = float(np.mean([p["qH_neg"] < 0 for p in neg])) if neg else float("nan")
        both = [p for p in pr if p.get("qH_s", 0) > 0 and p.get("qB_s", 0) > 0]
        out["qB_over_qH_median"] = float(np.median([p["qB_s"] / p["qH_s"] for p in both])) if both else float("nan")
    return out


def agg(vals):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)


def summary(runs):
    rows = []
    for bs in BS:
        for c in ORDER:
            if (c, bs) not in runs:
                continue
            ms = [run_metrics(L) for L in runs[(c, bs)]]
            row = {"config": c, "bs": bs, "seeds": len(ms)}
            for k in ms[0]:
                mean, std, n = agg([m.get(k) for m in ms])
                row[k] = mean
                row[k + "_std"] = std
            row["ep_to_97_reached"] = sum(1 for m in ms if not math.isnan(m["ep_to_97"]))
            rows.append(row)
    return rows


def fmt(x, p=2, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if pct:
        return f"{100 * x:.{p}f}"
    if abs(x) >= 1e4 or (abs(x) < 1e-2 and x != 0):
        return f"{x:.1e}"
    return f"{x:.{p}f}"


def write_tables(rows):
    import csv

    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    lines = ["| bs | method | test acc % (mean±sd) | train loss | epochs→97% | accept % | median μ | max μ | "
             "B indefinite % | median cond(M⁻¹) | max abs eig(RMRᵀ) | SR1 skip % | B-neg dirs truly neg % | grad evals | time s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        arcs = "accept" in r and not math.isnan(r.get("accept", float("nan")))
        e97 = fmt(r["ep_to_97"], 1) + ("" if r["ep_to_97_reached"] == r["seeds"] else f" ({r['ep_to_97_reached']}/{r['seeds']})")
        lines.append("| " + " | ".join([
            str(r["bs"]), LABEL[r["config"]],
            f"{100 * r['final_acc']:.2f} ± {100 * r['final_acc_std']:.2f}",
            fmt(r["final_train_loss"], 3), e97,
            fmt(r.get("accept"), 1, True) if arcs else "", fmt(r.get("mu_median")) if arcs else "",
            fmt(r.get("mu_max")) if arcs else "", fmt(r.get("frac_indef"), 1, True) if arcs else "",
            fmt(r.get("cond_minv_median")) if arcs else "", fmt(r.get("eig_max_abs_max")) if arcs else "",
            fmt(r.get("skip_rate"), 1, True) if arcs else "", fmt(r.get("neg_real_frac"), 0, True) if arcs else "",
            f"{r['n_grad']:.0f}", f"{r['time']:.0f}",
        ]) + " |")
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def mean_curve(logs, key):
    ys = [np.array([e[key] for e in L["epochs"]]) for L in logs]
    n = min(len(y) for y in ys)
    ys = np.stack([y[:n] for y in ys])
    return np.arange(n), ys.mean(0), ys.min(0), ys.max(0)


def smooth_log(v, w):
    v = np.log10(np.maximum(np.asarray(v, float), 1e-12))
    if len(v) < w:
        return 10 ** v
    k = np.ones(w) / w
    return 10 ** np.convolve(v, k, mode="valid")


def fig_accuracy(runs, configs, fname, title):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, bs in zip(axes, BS):
        for c in configs:
            if (c, bs) not in runs:
                continue
            x, m, lo, hi = mean_curve(runs[(c, bs)], "test_acc")
            ls = "--" if c in ("sgd", "adam") else "-"
            err, elo, ehi = 100 * (1 - m), 100 * (1 - hi), 100 * (1 - lo)
            ax.plot(x[1:], err[1:], ls, color=C[c], label=LABEL[c], marker="o", ms=3.5)
            ax.fill_between(x[1:], elo[1:], ehi[1:], color=C[c], alpha=0.12, lw=0)
        ax.set_title(f"batch size {bs}")
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.set_yticks([1.5, 2, 3, 5, 10, 20, 40])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(1.4, 50)
    axes[0].set_ylabel("test error (%), log scale")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(3, len(lab)), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.1 + 0.04 * (len(lab) > 3), 1, 0.97))
    fig.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close(fig)


def fig_mu(runs, configs, fname, title):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, bs in zip(axes, BS):
        for c in configs:
            if (c, bs) not in runs:
                continue
            L = sorted(runs[(c, bs)], key=lambda r: r["args"]["seed"])[0]
            mu = [d["mu"] for d in L["iters"]]
            w = max(5, len(mu) // 200)
            y = smooth_log(mu, w)
            ax.plot(np.arange(len(y)) + w // 2, y, color=C[c], label=LABEL[c], lw=1.6)
        ax.set_yscale("log")
        ax.set_title(f"batch size {bs}")
        ax.set_xlabel("iteration")
    axes[0].set_ylabel("cubic regularization μ (log, smoothed)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(3, len(lab)), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.13, 1, 0.97))
    fig.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close(fig)


def fig_curvature(runs, fname):
    """Does the negative curvature B reports exist in a 4096-sample Hessian?"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=False)
    for ax, bs in zip(axes, BS):
        any_pts = False
        for c in ["paper_cross", "safe_cross"]:
            pts = []
            for L in runs.get((c, bs), []):
                pts += [(p["lamB_neg"], p["qH_neg"]) for p in L["probes"] if "lamB_neg" in p]
            if not pts:
                continue
            a = np.array(pts)
            any_pts = True
            frac = np.mean(a[:, 1] < 0)
            ax.scatter(-a[:, 0], a[:, 1], s=14, color=C[c], alpha=0.55, lw=0,
                       label=f"{LABEL[c]}")
            ax.text(0.98, 0.04 + 0.08 * (c == "safe_cross"), ("γ = ½λ̂_min" if c == "safe_cross" else "γ = λ̂_min") + f":  n={len(a)}, truly negative {100 * frac:.0f}%",
                    transform=ax.transAxes, ha="right", fontsize=8.5, color="#0b0b0b")
            ax.scatter([], [])  # keep color order stable
        if not any_pts:
            ax.text(0.5, 0.5, "B never indefinite\nin these runs", transform=ax.transAxes, ha="center",
                    va="center", color="#52514e")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axhline(0, color="#52514e", lw=1)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1e-2)
        ax.set_title(f"batch size {bs}")
        ax.set_xlabel("|most negative eigenvalue of B|  (what SR1 claims)")
    axes[0].set_ylabel("uᵀHu on 4,096-sample Hessian\n(< 0 means the negative curvature is real)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Cross-batch y: negative curvature reported by B vs. a 4,096-sample Hessian along the same "
                 "direction (same-batch runs: B never indefinite)", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close(fig)


def fig_directions(rows, fname):
    configs = ["safe_cross", "safe_cross_layer", "safe_same", "safe_same_layer"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    for ax, bs in zip(axes, BS):
        rr = {r["config"]: r for r in rows if r["bs"] == bs}
        ys = np.arange(len(configs))[::-1]
        for y, c in zip(ys, configs):
            if c not in rr:
                continue
            r = rr[c]
            e = 100 * (1 - r["final_acc"])
            ax.errorbar(e, y, xerr=100 * r["final_acc_std"], fmt="o", color=C[c], ms=7, capsize=3, lw=1.5)
        for b in ("sgd", "adam"):
            if b in rr:
                ax.axvline(100 * (1 - rr[b]["final_acc"]), color=C[b], ls="--", lw=1.2, label=LABEL[b])
        ax.set_xscale("log")
        ax.set_xticks([1.5, 2, 3, 5, 10, 20, 40])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks(ys)
        ax.set_yticklabels([LABEL[c] for c in configs] if bs == BS[0] else [""] * len(configs), fontsize=8.5)
        ax.set_title(f"batch size {bs}")
        ax.set_xlabel("test error after 10 epochs (%), log")
        ax.set_ylim(-0.7, len(configs) - 0.3)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Per-layer subproblems (Dir. 1) and same-batch pairs (Dir. 3)  (mean ± sd over 3 seeds)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close(fig)


def main():
    runs = load()
    rows = summary(runs)
    print(write_tables(rows))
    fig_accuracy(runs, ["sgd", "adam", "paper_cross", "safe_cross", "safe_same"], "fig1_accuracy.png",
                 "Test error: cross-batch y (paper-style loop) vs. same-batch (s, y)  (mean, band = min–max over 3 seeds)")
    fig_mu(runs, ["paper_cross", "safe_cross", "bb_cross", "safe_same"], "fig2_mu_vs_gamma.png",
           "μ trajectories: the choice of γ (= δ in B₀ = γI) decides whether μ stays bounded  (seed 0)")
    fig_curvature(runs, "fig3_negative_curvature.png")
    fig_directions(rows, "fig4_directions.png")
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
