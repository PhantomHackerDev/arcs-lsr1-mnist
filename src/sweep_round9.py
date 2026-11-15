"""Round 9: larger batches (F-MNIST CNN, batch 1024 / 2048 / 4096).

ARCs-LSR1 (full network, same-batch y, m=10, 10 epochs = about 20 full-gradient
evaluations): floor mu_min in {none, 1, 4, 16, 64, 256} x eta2 in {0.75, 0.5, 0.25}
(eta1 = 0.1), with and without curvature pairs (B = I).
Baselines (20 epochs = the same compute as 10 ARCs epochs): SGD+momentum, AdamW,
Muon, Shampoo, SophiaG with a learning-rate sweep, and the paper's Adam (lr 1e-3).

Selection uses seed-0 final TRAINING loss only (the test set is not used):
  * baselines: the lr is chosen twice, at epoch 10 (same epochs as ARCs) and at
    epoch 20 (same compute); the sweep is extended by x3 or /3 until both are interior;
  * ARCs: the best (eta2, mu_min) per batch size, separately with and without pairs;
    if the best mu_min is the largest tried, the next x4 is tried.
Then seeds 1 and 2 at every selected configuration.

  python src/sweep_round9.py --workers 13   -> results/round9_selection.json
Skips runs whose JSON already exists, so it can be resumed.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid import C, FMNIST, ROUND8_FM, ROUND9_FM  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
RES = os.path.join(ROOT, "results")
BSS = (1024, 2048, 4096)
ARCS_EPOCHS, BASE_EPOCHS = 10, 20
LRS = {"sgd": (0.03, 0.1, 0.3), "adamw": (1e-3, 3e-3, 1e-2), "muon": (0.005, 0.015, 0.045),
       "shampoo": (1e-3, 3e-3, 1e-2), "sophia": (3e-4, 1e-3, 3e-3)}
FIXED = {"adam": 1e-3}  # the paper's Adam, not tuned
ETA2 = (0.75, 0.5, 0.25)


def arcs_name(pairs, e2, mu):
    """run name of one ARCs configuration (eta2 = 0.75 reuses the round-5 / round-8 names)"""
    p = "pairs" if pairs else "nopairs"
    vn = "0" if mu < 1e-3 else f"{mu:g}"
    if e2 == 0.75:
        return "fm_full_same" if (pairs and vn == "0") else f"r8_fm_{p}_mumin{vn}"
    return f"r9_fm_{p}_eta2_{e2:g}_mumin{vn}"


def arcs_args(pairs, e2, mu):
    name = arcs_name(pairs, e2, mu)
    if name in FMNIST:
        return FMNIST[name]
    if name in ROUND8_FM or name in ROUND9_FM:
        return {**ROUND8_FM, **ROUND9_FM}[name]
    pa = "" if pairs else "--no_pairs "  # extension beyond the grids
    return C + f"--opt arcs --y_mode same --probe_every 0 {pa}--eta2 {e2:g} --mu_min {mu:g}"


def base_name(opt, lr):
    return f"fm_{opt}_lr{lr:g}"


def job(name, args, bs, seed, epochs):
    tag = f"{name}_bs{bs}_s{seed}"
    cmd = (f"{PY} {ROOT}/src/run.py {args} --bs {bs} --seed {seed} --epochs {epochs} "
           f"--threads 1 --tag {tag} --outdir {RES}")
    # rough cost for scheduling: gradient evaluations x per-step overhead
    cost = epochs * (2 if "--opt arcs" in args else 1) * (1 + 60000 / bs / 30)
    return cost, tag, cmd


def run_jobs(jobs, workers):
    jobs = [j for j in jobs if not os.path.exists(os.path.join(RES, j[1] + ".json"))]
    jobs = sorted(set(jobs), reverse=True)  # longest first
    print(f"{len(jobs)} jobs", flush=True)

    def go(j):
        _, tag, cmd = j
        with open(os.path.join(RES, "logs", tag + ".log"), "w") as f:
            rc = subprocess.run(cmd.split(), stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"done {tag} rc={rc}", flush=True)
        return rc

    with ThreadPoolExecutor(workers) as ex:
        rcs = list(ex.map(go, jobs))
    print("failed:", sum(r != 0 for r in rcs), flush=True)


def final_loss(name, bs, epoch=-1, seed=0):
    f = os.path.join(RES, f"{name}_bs{bs}_s{seed}.json")
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        ep = json.load(fh)["epochs"]
    if epoch >= len(ep):  # run.py stops at the first non-finite loss
        return float("inf")
    tl = ep[epoch]["train_loss"]
    return tl if tl == tl else float("inf")  # NaN (diverged) loses


def tried_lrs(opt, bs):
    out = {}
    for f in glob.glob(os.path.join(RES, f"fm_{opt}_lr*_bs{bs}_s0.json")):
        m = re.match(rf"fm_{opt}_lr([\d.e+-]+)_bs{bs}_s0\.json$", os.path.basename(f))
        if m:
            lr = float(m.group(1))
            out[lr] = (final_loss(base_name(opt, lr), bs, ARCS_EPOCHS), final_loss(base_name(opt, lr), bs, BASE_EPOCHS))
    return out


def arcs_grid(bs, pairs):
    """{(eta2, mu): seed-0 final loss} over everything run so far for this batch size"""
    out = {}
    for e2 in ETA2:
        for mu in (1e-8, 1, 4, 16, 64, 256, 1024, 4096):
            tl = final_loss(arcs_name(pairs, e2, mu), bs)
            if tl is not None:
                out[(e2, mu)] = tl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=13)
    ap.add_argument("--max_ext", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(os.path.join(RES, "logs"), exist_ok=True)

    # 1. seed-0 grids
    jobs = []
    for bs in BSS:
        for pairs in (True, False):
            for e2 in ETA2:
                for mu in (1e-8, 1, 4, 16, 64, 256):
                    jobs.append(job(arcs_name(pairs, e2, mu), arcs_args(pairs, e2, mu), bs, 0, ARCS_EPOCHS))
        for opt, lrs in LRS.items():
            for lr in lrs:
                jobs.append(job(base_name(opt, lr), C + f"--opt {opt} --lr {lr:g}", bs, 0, BASE_EPOCHS))
        for opt, lr in FIXED.items():
            jobs.append(job(base_name(opt, lr), C + f"--opt {opt} --lr {lr:g}", bs, 0, BASE_EPOCHS))
    run_jobs(jobs, a.workers)

    # 2. extend until the selections are interior
    ext = {}
    while True:
        jobs = []
        for bs in BSS:
            for opt in LRS:
                tl = tried_lrs(opt, bs)
                lrs = sorted(tl)
                new = set()
                for i in (0, 1):  # epoch 10, epoch 20
                    best = min(lrs, key=lambda lr: tl[lr][i])
                    if best == lrs[0]:
                        new.add(("down", best / 3))
                    elif best == lrs[-1]:
                        new.add(("up", best * 3))
                for d, lr in new:
                    k = (bs, opt, d)
                    if ext.get(k, 0) < a.max_ext:
                        ext[k] = ext.get(k, 0) + 1
                        jobs.append(job(base_name(opt, lr), C + f"--opt {opt} --lr {lr:g}", bs, 0, BASE_EPOCHS))
            for pairs in (True, False):
                g = arcs_grid(bs, pairs)
                e2, mu = min(g, key=g.get)
                mus = sorted(m for (e, m) in g if e == e2)
                k = (bs, pairs, "mu")
                if mu == mus[-1] and ext.get(k, 0) < a.max_ext:
                    ext[k] = ext.get(k, 0) + 1
                    jobs.append(job(arcs_name(pairs, e2, mu * 4), arcs_args(pairs, e2, mu * 4), bs, 0, ARCS_EPOCHS))
        if not jobs:
            break
        print("extending:", [j[1] for j in jobs], flush=True)
        run_jobs(jobs, a.workers)

    # 3. selection and seeds 1-2
    sel, jobs = {}, []
    for bs in BSS:
        for opt in LRS:
            tl = tried_lrs(opt, bs)
            lrs = sorted(tl)
            b10 = min(lrs, key=lambda lr: tl[lr][0])
            b20 = min(lrs, key=lambda lr: tl[lr][1])
            sel[f"fm_bs{bs}_{opt}"] = {
                "lr_epoch10": b10, "lr_epoch20": b20,
                "tried": {f"{lr:g}": {"epoch10": tl[lr][0], "epoch20": tl[lr][1]} for lr in lrs},
                "at_edge": any(b in (lrs[0], lrs[-1]) for b in (b10, b20))}
            for lr in {b10, b20}:
                jobs += [job(base_name(opt, lr), C + f"--opt {opt} --lr {lr:g}", bs, s, BASE_EPOCHS) for s in (1, 2)]
        for opt, lr in FIXED.items():
            jobs += [job(base_name(opt, lr), C + f"--opt {opt} --lr {lr:g}", bs, s, BASE_EPOCHS) for s in (1, 2)]
        for pairs in (True, False):
            g = arcs_grid(bs, pairs)
            e2, mu = min(g, key=g.get)
            sel[f"fm_bs{bs}_arcs_{'pairs' if pairs else 'nopairs'}"] = {
                "eta2": e2, "mu_min": mu, "name": arcs_name(pairs, e2, mu), "train_loss_seed0": g[(e2, mu)],
                "tried": {f"eta2={e:g},mu_min={m:g}": v for (e, m), v in sorted(g.items())}}
            jobs += [job(arcs_name(pairs, e2, mu), arcs_args(pairs, e2, mu), bs, s, ARCS_EPOCHS) for s in (1, 2)]
    with open(os.path.join(RES, "round9_selection.json"), "w") as f:
        json.dump(sel, f, indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in v if kk != "tried"} for k, v in sel.items()}), flush=True)
    run_jobs(jobs, a.workers)


if __name__ == "__main__":
    main()
