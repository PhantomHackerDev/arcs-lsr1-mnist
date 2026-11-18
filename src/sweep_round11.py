"""Round 11: the paper's delta rule in the current setting (Aditya's question).

F-MNIST CNN, ARCs-LSR1 full network, same-batch y, 10 epochs, at the round-9 (eta2, mu_min) per batch size,
batch 1024 / 2048 / 4096, m in {10, 50, 100}, 3 seeds:
  * 'paper'  : delta = lambda_hat_min (Algorithm 1, step 4, as written), clipped to [1e-4, 1e4];
  * 'lm0.9'  : delta = 0.9 * lambda_hat_min, and the LM-damped handling of lambda_hat_min <= 0 ('safe_lm');
  * 'lm0.25', 'lm0.1': the same with 0.25 and 0.1 (m = 10 and 50 only);
  * reference: delta = 0.5 * lambda_hat_min ('safe_lm' / 'safe_keep', rounds 9-10; identical to the paper's
    text rule 'safe' whenever lambda_hat_min > 0). Only m = 10 at batch 2048 needs new runs.

  python src/sweep_round11.py --workers 13
Skips runs whose JSON already exists, so it can be resumed.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_round9 import RES, arcs_args, job, run_jobs  # noqa: E402
from sweep_round10 import setting  # noqa: E402

SEEDS = (0, 1, 2)
RULES = {"paper": "--delta_mode paper", "lm0.9": "--delta_mode safe_lm --safe_frac 0.9",
         "lm0.5": "--delta_mode safe_lm", "lm0.25": "--delta_mode safe_lm --safe_frac 0.25",
         "lm0.1": "--delta_mode safe_lm --safe_frac 0.1"}


def name(sel, bs, m, rule):
    e2, mu = setting(sel, bs)
    return f"r11_fm_pairs_eta2_{e2:g}_mumin{mu:g}_m{m}_{rule}"


def args(sel, bs, m, rule):
    e2, mu = setting(sel, bs)
    return arcs_args(True, e2, mu) + f" --history {m} " + RULES[rule]


def all_jobs():
    jobs = []
    for bs in (1024, 2048, 4096):
        for m in (10, 50, 100):
            for rule in ("paper", "lm0.9"):
                jobs += [(bs, m, rule, s) for s in SEEDS]
    jobs += [(2048, 10, "lm0.5", s) for s in SEEDS]
    # extension: is 0.5 the best fraction? (smaller fractions, m = 10 and 50)
    for bs in (1024, 2048, 4096):
        for m in (10, 50):
            for rule in ("lm0.25", "lm0.1"):
                jobs += [(bs, m, rule, s) for s in SEEDS]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=13)
    a = ap.parse_args()
    with open(os.path.join(RES, "round9_selection.json")) as f:
        sel = json.load(f)
    run_jobs([job(name(sel, bs, m, r), args(sel, bs, m, r), bs, s, 10) for bs, m, r, s in all_jobs()], a.workers)


if __name__ == "__main__":
    main()
