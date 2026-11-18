"""Round 13: the delta rule of the authors' code ('code') where lambda_hat_min <= 0 actually happens.

delta = max(0.5 lambda_hat_min, 1e-6) if lambda_hat_min > 0, else min(1.5 lambda_hat_min, -1e-6) (negative delta).
When lambda_hat_min > 0 this is 'safe' (rounds 2-12) up to the clipping bound, so the comparison is only
informative where lambda_hat_min <= 0 occurs: batch 2048 with m = 20 / 50 / 100 / 200, and double_grad at batch
1024 and 4096 (m = 50). F-MNIST CNN, round-9 (eta2, mu_min), 10 epochs, 3 seeds. The references
('safe', 'safe_keep', 'safe_lm') are the runs of rounds 9-10.

  python src/sweep_round13.py --workers 13
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_round9 import RES, arcs_args, job, run_jobs  # noqa: E402
from sweep_round10 import DG, setting  # noqa: E402

SEEDS = (0, 1, 2)
CONFIGS = [(2048, 20, ""), (2048, 50, ""), (2048, 100, ""), (2048, 200, ""), (1024, 50, "_dg"), (4096, 50, "_dg")]


def name(sel, bs, m, dg):
    e2, mu = setting(sel, bs)
    return f"r13_fm_pairs_eta2_{e2:g}_mumin{mu:g}_m{m}_code{dg}"


def args(sel, bs, m, dg):
    e2, mu = setting(sel, bs)
    return arcs_args(True, e2, mu) + f" --history {m} --delta_mode code" + DG[dg]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=13)
    a = ap.parse_args()
    with open(os.path.join(RES, "round9_selection.json")) as f:
        sel = json.load(f)
    run_jobs([job(name(sel, bs, m, dg), args(sel, bs, m, dg), bs, s, 10) for bs, m, dg in CONFIGS for s in SEEDS],
             a.workers)


if __name__ == "__main__":
    main()
