"""Round 6: modern baselines with an adaptive learning-rate sweep.

For each setting and optimizer: start from the seed-0 runs of the initial grid
(grid.py --suite modern_*), pick the lr with the lowest final TRAINING loss
(the test set is not used for tuning). If that lr is at an edge of what has
been tried, try lr/3 or lr*3 (at most `max_ext` times per direction) until
the best lr is interior. Then run seeds 1 and 2 at the chosen lr.

  python src/sweep_round6.py --workers 14   -> results/round6_selection.json
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SETTINGS = [  # (tag prefix, run.py args, batch size)
    ("deep", "--arch deep --data mnist --probe_every 100", 256),
    ("fm", "--arch cnn --data fmnist --probe_every 100", 64),
    ("fm", "--arch cnn --data fmnist --probe_every 100", 256),
]
OPTS = ("adamw", "muon", "shampoo", "sophia")


def tag(prefix, opt, lr, bs, seed):
    return f"{prefix}_{opt}_lr{lr:g}_bs{bs}_s{seed}"


def tried(prefix, opt, bs):
    """{lr: final train loss} of the seed-0 runs so far (NaN/inf counts as worst)."""
    out = {}
    for f in glob.glob(os.path.join(ROOT, "results", f"{prefix}_{opt}_lr*_bs{bs}_s0.json")):
        m = re.match(rf"{prefix}_{opt}_lr([\d.e+-]+)_bs{bs}_s0\.json$", os.path.basename(f))
        if not m:
            continue
        with open(f) as fh:
            tl = json.load(fh)["epochs"][-1]["train_loss"]
        out[float(m.group(1))] = tl if tl == tl else float("inf")
    return out


def run_jobs(jobs, workers, threads, epochs):
    def go(j):
        prefix, args, bs, opt, lr, seed = j
        t = tag(prefix, opt, lr, bs, seed)
        if os.path.exists(os.path.join(ROOT, "results", t + ".json")):
            return 0
        cmd = (f"{PY} {ROOT}/src/run.py {args} --opt {opt} --lr {lr:g} --bs {bs} --seed {seed} "
               f"--epochs {epochs} --threads {threads} --tag {t} --outdir {ROOT}/results")
        with open(os.path.join(ROOT, "results", "logs", t + ".log"), "w") as f:
            rc = subprocess.run(cmd.split(), stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"done {t} rc={rc}", flush=True)
        return rc

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(go, jobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max_ext", type=int, default=2)
    a = ap.parse_args()
    ext = {}  # (prefix, bs, opt) -> extensions done
    while True:
        jobs = []
        for prefix, args, bs in SETTINGS:
            for opt in OPTS:
                tl = tried(prefix, opt, bs)
                if not tl:
                    continue
                best = min(tl, key=tl.get)
                k = (prefix, bs, opt)
                if ext.get(k, 0) >= a.max_ext:
                    continue
                if best == min(tl):
                    new = best / 3
                elif best == max(tl):
                    new = best * 3
                else:
                    continue
                ext[k] = ext.get(k, 0) + 1
                jobs.append((prefix, args, bs, opt, new, 0))
        if not jobs:
            break
        print(f"extending sweep: {[(j[0], j[2], j[3], f'{j[4]:g}') for j in jobs]}", flush=True)
        run_jobs(jobs, a.workers, a.threads, a.epochs)

    selection, jobs = {}, []
    for prefix, args, bs in SETTINGS:
        for opt in OPTS:
            tl = tried(prefix, opt, bs)
            if not tl:
                continue
            best = min(tl, key=tl.get)
            lrs = sorted(tl)
            selection[f"{prefix}_bs{bs}_{opt}"] = {
                "lr": best, "train_loss_seed0": tl[best], "tried": {f"{lr:g}": tl[lr] for lr in lrs},
                "at_edge": best in (lrs[0], lrs[-1])}
            jobs += [(prefix, args, bs, opt, best, s) for s in (1, 2)]
    with open(os.path.join(ROOT, "results", "round6_selection.json"), "w") as f:
        json.dump(selection, f, indent=1)
    print(json.dumps({k: (v["lr"], v["at_edge"]) for k, v in selection.items()}), flush=True)
    rcs = run_jobs(jobs, a.workers, a.threads, a.epochs)
    print("failed:", sum(r != 0 for r in rcs), flush=True)


if __name__ == "__main__":
    main()
