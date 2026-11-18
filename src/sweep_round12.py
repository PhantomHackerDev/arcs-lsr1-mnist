"""Round 12: diagonal initial matrix B0 = delta * D (--diag_power p), F-MNIST CNN, 10 epochs, 3 seeds.

D = (sqrt(v_hat) + eps)^p normalized to mean 1, v the EMA of g^2 (beta2 = 0.999), refreshed every step; p = 0.5
and 1. Each batch size uses its best configuration of rounds 9-10 (delta rule 'safe_lm'), which is the reference
(p = 0):
  batch 256 / 1024: m = 50 + double_grad;  batch 2048 / 4096: m = 100.

  python src/sweep_round12.py --workers 13
Skips runs whose JSON already exists, so it can be resumed.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_round9 import RES, job, run_jobs  # noqa: E402
from sweep_round10 import args as args10, name as name10  # noqa: E402

SEEDS = (0, 1, 2)
CONFIGS = [(256, 50, "_dg"), (1024, 50, "_dg"), (2048, 100, ""), (4096, 100, "")]
POWERS = (0.5, 1.0)


def name(sel, bs, m, dg, p):
    return name10(sel, bs, m, "lm", dg).replace("r9m_", "r12_").replace("r10_", "r12_") + f"_diag{p:g}"


def args(sel, bs, m, dg, p):
    return args10(sel, bs, m, "lm", dg) + f" --diag_power {p:g}"


MU_CHECK = (0.25, 4.0)  # follow-up: is the round-9 mu_min still right with D? (batch 1024 and 4096, p = 1)


def all_jobs(sel):
    jobs = [job(name(sel, bs, m, dg, p), args(sel, bs, m, dg, p), bs, s, 10)
            for bs, m, dg in CONFIGS for p in POWERS for s in SEEDS]
    for bs, m, dg in CONFIGS:
        if bs in (1024, 4096):
            for mu in MU_CHECK:
                n = name(sel, bs, m, dg, 1.0).replace("_mumin1_", f"_mumin{mu:g}_")
                a = args(sel, bs, m, dg, 1.0).replace("--mu_min 1 ", f"--mu_min {mu:g} ")
                assert f"--mu_min {mu:g} " in a
                jobs += [job(n, a, bs, s, 10) for s in SEEDS]
    # follow-up 2: mu_min = 0.25 was best (an edge), so 0.0625 too; and the reference (p = 0) at the lower floors,
    # so that a gain of the diagonal B0 is not just the lower floor
    for bs, m, dg in CONFIGS:
        if bs in (1024, 4096):
            for p, mus in ((1.0, (0.0625,)), (0.0, (0.25, 0.0625))):
                for mu in mus:
                    jobs += [job(name_mu(sel, bs, m, dg, p, mu), args_mu(sel, bs, m, dg, p, mu), bs, s, 10)
                             for s in SEEDS]
    return jobs


def name_mu(sel, bs, m, dg, p, mu):
    """run name with mu_min replaced; p = 0 is the reference without a diagonal B0"""
    n = name(sel, bs, m, dg, 1.0).replace("_mumin1_", f"_mumin{mu:g}_")
    return n.replace("_diag1", "") if p == 0 else n.replace("_diag1", f"_diag{p:g}")


def args_mu(sel, bs, m, dg, p, mu):
    a = args10(sel, bs, m, "lm", dg).replace("--mu_min 1 ", f"--mu_min {mu:g} ")
    assert f"--mu_min {mu:g} " in a
    return a + (f" --diag_power {p:g}" if p else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=13)
    a = ap.parse_args()
    with open(os.path.join(RES, "round9_selection.json")) as f:
        sel = json.load(f)
    run_jobs(all_jobs(sel), a.workers)


if __name__ == "__main__":
    main()
