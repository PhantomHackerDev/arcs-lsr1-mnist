"""Single-precision L-SR1 state (--vec_dtype float32) vs. the float64 default, on the best round-10 configurations.

F-MNIST CNN, 10 epochs, 3 seeds. The float64 runs are the existing round-9/10 runs; the float32 runs repeat them
with --vec_dtype float32 (the [S|Y] buffer and all parameter-sized vectors in single precision, the small
2m x 2m linear algebra still in float64).

  python src/check_fp32.py --workers 12        -> runs, then results/summary_fp32.md
  python src/check_fp32.py --analyze           -> only the summary
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_round9 import at, fmt, load  # noqa: E402
from sweep_round9 import RES, job, run_jobs  # noqa: E402
from sweep_round10 import args as args10, name as name10  # noqa: E402

SEEDS = (0, 1, 2)
CONFIGS = [(256, 50, "lm", "_dg"), (1024, 50, "lm", "_dg"), (2048, 100, "lm", ""), (4096, 100, "keep", "")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    with open(os.path.join(RES, "round9_selection.json")) as f:
        sel = json.load(f)
    if not a.analyze:
        jobs = []
        for bs, m, rule, dg in CONFIGS:
            n32 = name10(sel, bs, m, rule, dg).replace("r9m_", "r10_") + "_f32"
            jobs += [job(n32, args10(sel, bs, m, rule, dg) + " --vec_dtype float32", bs, s, 10) for s in SEEDS]
        run_jobs(jobs, a.workers)
    lines = ["### Single-precision L-SR1 state vs. float64 (F-MNIST CNN, 10 epochs, 3 seeds)", "",
             "| batch, m, δ rule | float64: train loss | float32: train loss | float64 / float32 test acc % | "
             "max |Δ loss| over seeds | wall time per run, float64 / float32 (s) |", "|---|---|---|---|---|---|"]
    for bs, m, rule, dg in CONFIGS:
        n64 = name10(sel, bs, m, rule, dg)
        n32 = n64.replace("r9m_", "r10_") + "_f32"
        L64, L32 = load(n64, bs), load(n32, bs)
        if not (L64 and L32):
            continue
        d = max(abs(x["epochs"][-1]["train_loss"] - y["epochs"][-1]["train_loss"]) for x, y in zip(L64, L32))
        t64 = np.mean([L["epochs"][-1]["time"] for L in L64])
        t32 = np.mean([L["epochs"][-1]["time"] for L in L32])
        lines.append(f"| {bs}, m = {m}, safe_{rule}{' + double_grad' if dg else ''} | {fmt(*at(L64, 10))} | "
                     f"{fmt(*at(L32, 10))} | {100 * at(L64, 10, 'test_acc')[0]:.2f} / {100 * at(L32, 10, 'test_acc')[0]:.2f}"
                     f" | {d:.4f} | {t64:.0f} / {t32:.0f} |")
    lines += ["", "Wall times are from runs sharing the machine (one thread each), so they are only indicative.", ""]
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(RES, "summary_fp32.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
