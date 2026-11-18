"""Round 13: the delta rule of the authors' code vs. 'safe' / 'safe_keep' / 'safe_lm', where lambda_hat_min <= 0.

  python src/analyze_round13.py -> results/summary_round13.md
"""
import json
import os

import numpy as np

from analyze import RES  # noqa: E402
from analyze_round9 import at, fmt, lam_neg, load  # noqa: E402
from sweep_round9m import name as name9  # noqa: E402
from sweep_round10 import name as name10  # noqa: E402
from sweep_round13 import CONFIGS, name as name13  # noqa: E402

with open(os.path.join(RES, "round9_selection.json")) as f:
    SEL = json.load(f)


def runs(bs, m, dg, rule):
    if rule == "code":
        return load(name13(SEL, bs, m, dg), bs)
    if rule == "safe":
        return load(name9(SEL, bs, m, False), bs) if (not dg and m in (5, 20, 50)) else []
    if rule == "keep" and not dg and m in (10, 20, 50, 100):
        return load(name9(SEL, bs, m, True), bs)
    return load(name10(SEL, bs, m, rule, dg), bs)


def col(logs, key):
    return np.array([np.ravel(d[key])[0] for L in logs for d in L["iters"] if key in d], dtype=float)


LABEL = {"safe": "safe: 0.5 λ̂_min, else δ = 1", "keep": "safe_keep: else previous δ",
         "lm": "safe_lm: else previous δ + LM damping", "code": "code: else δ = 1.5 λ̂_min < 0 (authors)"}


def main():
    lines = ["### Round 13: the δ rule of the authors' code (F-MNIST CNN, 10 epochs)", "",
             "All four rules use δ = 0.5 λ̂_min when λ̂_min > 0; they differ only when λ̂_min ≤ 0. "
             "Training loss after 10 epochs (3 seeds unless n is shown).", "",
             "| batch, m | δ rule (λ̂_min ≤ 0 case) | train loss | test acc % | accepted % | λ̂_min ≤ 0 (% of steps) "
             "| δ < 0 (% of steps) | median δ |", "|---|---|---|---|---|---|---|---|"]
    for bs, m, dg in CONFIGS:
        for rule in ("safe", "keep", "lm", "code"):
            logs = runs(bs, m, dg, rule)
            if not logs:
                continue
            dl = col(logs, "delta")
            acc = 100 * col(logs, "acc").mean()
            lines.append(f"| {bs}, m = {m}{' + double_grad' if dg else ''} | {LABEL[rule]} | {fmt(*at(logs, 10))} | "
                         f"{100 * at(logs, 10, 'test_acc')[0]:.2f} | {acc:.0f} | {lam_neg(logs):.0f} | "
                         f"{100 * np.mean(dl < 0):.0f} | {np.median(dl):.3g} |")
        lines.append("")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(RES, "summary_round13.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
