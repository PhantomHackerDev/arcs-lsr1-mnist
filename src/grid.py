"""Run the full experiment grid with a small process pool.

  python src/grid.py --workers 8 --threads 2 --epochs 10
Skips runs whose JSON already exists, so it can be resumed.
"""
import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

ARCS = {
    # gamma (= delta) strategy x how y is measured
    "paper_cross": "--delta_mode paper --y_mode cross",
    "safe_cross": "--delta_mode safe --y_mode cross",
    "bb_cross": "--delta_mode bb --y_mode cross",
    "paper_same": "--delta_mode paper --y_mode same",
    "safe_same": "--delta_mode safe --y_mode same",
    "bb_same": "--delta_mode bb --y_mode same",
    # Direction 1: per-layer subproblems
    "safe_cross_layer": "--delta_mode safe --y_mode cross --per_layer",
    "safe_same_layer": "--delta_mode safe --y_mode same --per_layer",
    # Direction 2: momentum on the model gradient
    "safe_cross_gb0.9": "--delta_mode safe --y_mode cross --grad_beta 0.9",
    "safe_same_gb0.9": "--delta_mode safe --y_mode same --grad_beta 0.9",
}
BASE = {"sgd": "--opt sgd", "adam": "--opt adam"}

# Follow-up on the deeper net (784-512-256-128-10): Aditya's two questions
D = "--arch deep --probe_every 100 "
LAYER = {  # Q1: one mu vs a mu_l per layer
    "deep_full_same": D + "--opt arcs --y_mode same",
    "deep_layer_same": D + "--opt arcs --y_mode same --per_layer",
    "deep_layermu_same": D + "--opt arcs --y_mode same --per_layer --mu_per_layer",
    "deep_full_cross": D + "--opt arcs --y_mode cross",
    "deep_layer_cross": D + "--opt arcs --y_mode cross --per_layer",
    "deep_layermu_cross": D + "--opt arcs --y_mode cross --per_layer --mu_per_layer",
    "deep_sgd": D + "--opt sgd",
    "deep_adam": D + "--opt adam",
}
LM = D + "--opt arcs --y_mode same --per_layer --mu_per_layer "
MEMORY = {  # Q2: can Hessian momentum replace history?  (m=10 is deep_layermu_same)
    "deep_layermu_same_m1": LM + "--history 1",
    "deep_layermu_same_m2": LM + "--history 2",
    "deep_layermu_same_m5": LM + "--history 5",
    "deep_layermu_same_m1_hb0.5": LM + "--history 1 --hess_beta 0.5 --rank_cap 2",
    "deep_layermu_same_m1_hb0.9": LM + "--history 1 --hess_beta 0.9 --rank_cap 2",
    "deep_layermu_same_m2_hb0.5": LM + "--history 2 --hess_beta 0.5 --rank_cap 4",
    "deep_layermu_same_m2_hb0.9": LM + "--history 2 --hess_beta 0.9 --rank_cap 4",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", default="64,256,1024")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--only", default="", help="comma list of config names")
    ap.add_argument("--suite", default="main", choices=["main", "layer", "memory"])
    a = ap.parse_args()
    outdir = os.path.join(ROOT, "results")
    logdir = os.path.join(ROOT, "results", "logs")
    os.makedirs(logdir, exist_ok=True)

    jobs = []
    if a.suite == "main":
        confs = {**{k: "--opt arcs " + v for k, v in ARCS.items()}, **BASE}
    elif a.suite == "layer":
        confs = LAYER
    else:
        confs = MEMORY
    if a.only:
        confs = {k: v for k, v in confs.items() if k in a.only.split(",")}
    for bs, (cname, cargs), seed in itertools.product(
            [int(b) for b in a.bs.split(",")], confs.items(), [int(s) for s in a.seeds.split(",")]):
        tag = f"{cname}_bs{bs}_s{seed}"
        if os.path.exists(os.path.join(outdir, tag + ".json")):
            continue
        cmd = (f"{PY} {ROOT}/src/run.py {cargs} --bs {bs} --seed {seed} --epochs {a.epochs} "
               f"--threads {a.threads} --tag {tag} --outdir {outdir}")
        cost = (60000 / bs) * (2 if "same" in cname else 1) * (0.2 if ("sgd" in cname or "adam" in cname) else 1)
        jobs.append((cost, tag, cmd))
    jobs.sort(reverse=True)  # longest first
    print(f"{len(jobs)} jobs", flush=True)

    def run(job):
        _, tag, cmd = job
        with open(os.path.join(logdir, tag + ".log"), "w") as f:
            r = subprocess.run(cmd.split(), stdout=f, stderr=subprocess.STDOUT)
        print(f"done {tag} rc={r.returncode}", flush=True)
        return r.returncode

    with ThreadPoolExecutor(a.workers) as ex:
        rcs = list(ex.map(run, jobs))
    print("failed:", sum(r != 0 for r in rcs), flush=True)


if __name__ == "__main__":
    main()
