"""Round 9, part 2: L-SR1 memory m at larger batches (F-MNIST CNN, batch 1024 / 2048 / 4096).

At each batch size, ARCs-LSR1 with the (eta2, mu_min) selected in round 9 (results/round9_selection.json)
and m in {5, 10, 20, 50, 100}:
  * seed 0 with the paper's delta rule 'safe' (m = 5, 20, 50; m = 10 is the round-9 run);
  * seeds 0-2 with 'safe_keep' (m = 10, 20, 50, 100): the same rule, except that when lambda_hat_min <= 0
    the previous delta is kept instead of resetting to delta_fallback = 1. With many pairs lambda_hat_min
    can stay negative (batch 2048, m = 50: 99.7% of the steps), and 'safe' then runs with delta = 1.

  python src/sweep_round9m.py --workers 13
Skips runs whose JSON already exists, so it can be resumed.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_round9 import BSS, RES, arcs_args, job, run_jobs  # noqa: E402

MS_SAFE = (5, 20, 50)
MS_KEEP = (10, 20, 50, 100)


def name(sel, bs, m, keep):
    s = sel[f"fm_bs{bs}_arcs_pairs"]
    return f"r9m_fm_pairs_eta2_{s['eta2']:g}_mumin{s['mu_min']:g}_m{m}" + ("_keep" if keep else "")


def args(sel, bs, m, keep):
    s = sel[f"fm_bs{bs}_arcs_pairs"]
    return arcs_args(True, s["eta2"], s["mu_min"]) + f" --history {m}" + (" --delta_mode safe_keep" if keep else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=13)
    a = ap.parse_args()
    with open(os.path.join(RES, "round9_selection.json")) as f:
        sel = json.load(f)
    jobs = []
    for bs in BSS:
        for m in MS_SAFE:
            jobs.append(job(name(sel, bs, m, False), args(sel, bs, m, False), bs, 0, 10))
        for m in MS_KEEP:
            for seed in (0, 1, 2):
                jobs.append(job(name(sel, bs, m, True), args(sel, bs, m, True), bs, seed, 10))
    run_jobs(jobs, a.workers)


if __name__ == "__main__":
    main()
