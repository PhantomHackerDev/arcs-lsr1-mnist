"""Round 10: two ideas from Goldfarb, Ren & Bahamou (2020), F-MNIST CNN, ARCs-LSR1 full network, same-batch y.

(a) LM damping of the pairs for the delta rule ('safe_lm'): when lambda_hat_min <= 0, y_i -> y_i + lam s_i
    with lam = delta_prev / safe_frac - lambda_hat_min, so delta = delta_prev and B >= delta I.
    Batch 2048 (where 'safe' / 'safe_keep' run into lambda_hat_min <= 0), m in {20, 50, 100}, 3 seeds;
    m = 200 at every batch size, 'safe_keep' vs 'safe_lm', seed 0.
(b) Gradient over two batches at no extra cost ('double_grad'): the model gradient averages the current batch's
    gradient with the previous batch's gradient at the same point (already computed for the same-batch pair).
    'double_grad_ratio' also measures the ratio on the two-batch average loss (one extra forward pass).
    m = 50, delta rule 'safe_keep', at the round-9 (eta2, mu_min) for batch 1024 / 2048 / 4096, and at round 8's
    (eta2 = 0.75, mu_min = 4) for batch 256 (with its own reference run); 3 seeds.
    Follow-up: 'double_grad' with 'safe_lm' (3 seeds), and m = 200 with 'safe_lm' (seeds 1-2).

  python src/sweep_round10.py --workers 13
Skips runs whose JSON already exists, so it can be resumed.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_round9 import RES, arcs_args, job, run_jobs  # noqa: E402

SEEDS = (0, 1, 2)
DG = {"": "", "_dg": " --double_grad", "_dgr": " --double_grad_ratio"}


def setting(sel, bs):
    """(eta2, mu_min) of the reference ARCs-LSR1 at this batch size"""
    if bs == 256:
        return 0.75, 4.0  # round 8 selection
    s = sel[f"fm_bs{bs}_arcs_pairs"]
    return s["eta2"], s["mu_min"]


def name(sel, bs, m, rule, dg=""):
    e2, mu = setting(sel, bs)
    if rule == "keep" and dg == "" and bs != 256 and m in (10, 20, 50, 100):  # round-9 run
        return f"r9m_fm_pairs_eta2_{e2:g}_mumin{mu:g}_m{m}_keep"
    return f"r10_fm_pairs_eta2_{e2:g}_mumin{mu:g}_m{m}_{rule}{dg}"


def args(sel, bs, m, rule, dg=""):
    e2, mu = setting(sel, bs)
    return arcs_args(True, e2, mu) + f" --history {m} --delta_mode safe_{rule}" + DG[dg]


def all_jobs(sel):
    jobs = []
    for m in (20, 50, 100):  # (a)
        jobs += [(2048, m, "lm", "", s) for s in SEEDS]
    for bs in (1024, 2048, 4096):
        jobs += [(bs, 200, r, "", 0) for r in ("keep", "lm")]
    for bs in (256, 1024, 2048, 4096):  # (b)
        for dg in ("", "_dg", "_dgr"):
            if dg or bs == 256:
                jobs += [(bs, 50, "keep", dg, s) for s in SEEDS]
    # follow-up: at batch >= 1024 double_grad came with lambda_hat_min <= 0 in 33-99% of the steps under
    # 'safe_keep', so (b) again with 'safe_lm'; and seeds 1-2 for m = 200 with 'safe_lm'.
    # ('safe_lm' == 'safe_keep' whenever lambda_hat_min > 0 throughout, e.g. m = 50 at batch 1024 / 4096.)
    for bs in (256, 1024, 2048, 4096):
        jobs += [(bs, 50, "lm", "_dg", s) for s in SEEDS]
    jobs += [(256, 50, "lm", "", s) for s in SEEDS]
    for bs in (1024, 2048, 4096):
        jobs += [(bs, 200, "lm", "", s) for s in (1, 2)]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=13)
    a = ap.parse_args()
    with open(os.path.join(RES, "round9_selection.json")) as f:
        sel = json.load(f)
    run_jobs([job(name(sel, bs, m, r, dg), args(sel, bs, m, r, dg), bs, s, 10) for bs, m, r, dg, s in all_jobs(sel)],
             a.workers)


if __name__ == "__main__":
    main()
