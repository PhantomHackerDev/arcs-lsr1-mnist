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

FIX = {  # round 3: fixes for the mu_l-per-layer failure with noisy pairs
    "deep_layermu_floor1_cross": LM.replace("--y_mode same", "--y_mode cross") + "--mu_floor_kappa 1",
    "deep_layermu_floor1_same": LM + "--mu_floor_kappa 1",
    "deep_layermu_next_cross": LM.replace("--y_mode same", "--y_mode cross") + "--rho_batch next",
    "deep_layermu_next_same": LM + "--rho_batch next",
}

# round 4: Aditya's suggestion to favour B_new (hess_beta = weight on the OLD model)
FLIP = {}
for m in (1, 2):
    for hb in (0.1, 0.3):  # weight on B_new = 0.9, 0.7
        for cap in (m, 2 * m):
            FLIP[f"deep_layermu_same_m{m}_hb{hb}_cap{cap}"] = LM + f"--history {m} --hess_beta {hb} --rank_cap {cap}"
# secant residual vs. weight on the old model (1 seed, short runs; hb 0 = plain L-SR1)
SECANT = {f"deep_secant_m2_hb{hb}": LM + f"--history 2 --hess_beta {hb} --rank_cap 4" if hb else LM + "--history 2"
          for hb in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)}

# round 5: the paper's F-MNIST CNN (21,840 params, 4 layer blocks)
C = "--arch cnn --data fmnist --probe_every 100 "
FMNIST = {
    "fm_sgd": C + "--opt sgd",
    "fm_adam": C + "--opt adam",
    "fm_full_same": C + "--opt arcs --y_mode same",
    "fm_full_cross": C + "--opt arcs --y_mode cross",
    "fm_layer_same": C + "--opt arcs --y_mode same --per_layer",
    "fm_layer_cross": C + "--opt arcs --y_mode cross --per_layer",
    "fm_layermu_floor_same": C + "--opt arcs --y_mode same --per_layer --mu_per_layer --mu_floor_kappa 1",
    "fm_layermu_floor_cross": C + "--opt arcs --y_mode cross --per_layer --mu_per_layer --mu_floor_kappa 1",
    "fm_layer_same_m2": C + "--opt arcs --y_mode same --per_layer --history 2",
    "fm_layermu_floor_same_m2": C + "--opt arcs --y_mode same --per_layer --mu_per_layer --mu_floor_kappa 1 --history 2",
}

# round 6: modern baselines with a learning-rate sweep (seed 0), then 2 more seeds at the best lr
LRS = {"adamw": (3e-4, 1e-3, 3e-3), "muon": (0.005, 0.02, 0.05), "shampoo": (3e-4, 1e-3, 3e-3),
       "sophia": (3e-4, 1e-3, 3e-3)}
MODERN_MNIST = {f"deep_{o}_lr{lr:g}": D + f"--opt {o} --lr {lr:g}" for o, lrs in LRS.items() for lr in lrs}
# Round 7 (F-MNIST CNN, full network, same-batch y, m=10): Aditya's go-ahead on
#  (R) the ratio test on a larger batch (the current batch + random others),
#  (Y) curvature pairs and ratio test on that larger batch,
#  (H) tuning the ARCs hyperparameters one at a time from the defaults.
A7 = C + "--opt arcs --y_mode same --probe_every 0 "
ETAS = {"eta0.1-0.75": "--eta1 0.1 --eta2 0.75", "eta0.01-0.25": "--eta1 0.01 --eta2 0.25",
        "eta0-0.1": "--eta1 0 --eta2 0.1"}
ROUND7 = {}
for nb in (256, 1024):
    for en, ea in ETAS.items():
        ROUND7[f"r7_rho{nb}_{en}"] = A7 + f"--rho_bs {nb} {ea}"
        if en != "eta0.1-0.75":
            ROUND7[f"r7_yrho{nb}_{en}"] = A7 + f"--rho_bs {nb} --y_on_rho {ea}"
for v in (0, 0.01, 0.25, 0.5):
    ROUND7[f"r7_eta1_{v:g}"] = A7 + f"--eta1 {v:g}"
for v in (0.25, 0.5, 0.9):
    ROUND7[f"r7_eta2_{v:g}"] = A7 + f"--eta2 {v:g}"
ROUND7["r7_eta0.01-0.25"] = A7 + ETAS["eta0.01-0.25"]
ROUND7["r7_eta0-0.1"] = A7 + ETAS["eta0-0.1"]
for v in (0.01, 100):
    ROUND7[f"r7_mu0_{v:g}"] = A7 + f"--mu0 {v:g}"
for v in (1, 4, 16):
    ROUND7[f"r7_mumin_{v:g}"] = A7 + f"--mu_min {v:g}"
ROUND7["r7_bb"] = A7 + "--delta_mode bb"
# extension: mu_min and eta1 were best at the edge of the first range; combinations
for v in (64, 256):
    ROUND7[f"r7_mumin_{v:g}"] = A7 + f"--mu_min {v:g}"
ROUND7["r7_eta1_0.7"] = A7 + "--eta1 0.7"
for v in (4, 16, 64):  # ablation: same mu floor, no L-SR1 curvature (B = I)
    ROUND7[f"r7_nopairs_mumin_{v:g}"] = A7 + f"--no_pairs --mu_min {v:g}"
ROUND7["r7_mumin16_eta1_0.5"] = A7 + "--mu_min 16 --eta1 0.5"
ROUND7["r7_mumin16_rho256_eta0.01-0.25"] = A7 + "--mu_min 16 --rho_bs 256 " + ETAS["eta0.01-0.25"]
ROUND7["r7_mumin16_yrho256_eta0.01-0.25"] = A7 + "--mu_min 16 --rho_bs 256 --y_on_rho " + ETAS["eta0.01-0.25"]

# Round 8: does the L-SR1 curvature help when the gradient is less noisy?
# mu_min sweep with and without curvature pairs (B = I), full network, same-batch y.
# mu_min 1e-8 (the default) with pairs is round 5's fm_full_same / round 2's deep_full_same.
def _r8(base, name):
    out = {}
    for v in (1e-8, 1, 4, 16, 64, 256):
        vn = "0" if v < 1e-3 else f"{v:g}"
        for pn, pa in (("pairs", ""), ("nopairs", "--no_pairs ")):
            if v < 1e-3 and pn == "pairs":
                continue  # existing run
            out[f"r8_{name}_{pn}_mumin{vn}"] = base + f"--opt arcs --y_mode same --probe_every 0 {pa}--mu_min {v:g}"
    return out


ROUND8_FM = _r8(C, "fm")
ROUND8_DEEP = _r8(D, "deep")

MODERN_FMNIST = {f"fm_{o}_lr{lr:g}": C + f"--opt {o} --lr {lr:g}" for o, lrs in LRS.items() for lr in lrs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", default="64,256,1024")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--only", default="", help="comma list of config names")
    ap.add_argument("--suite", default="main", choices=["main", "layer", "memory", "fix", "flip", "secant", "fmnist",
                                                   "modern_mnist", "modern_fmnist", "round7",
                                                   "round8_fm", "round8_deep"])
    a = ap.parse_args()
    outdir = os.path.join(ROOT, "results")
    logdir = os.path.join(ROOT, "results", "logs")
    os.makedirs(logdir, exist_ok=True)

    jobs = []
    if a.suite == "main":
        confs = {**{k: "--opt arcs " + v for k, v in ARCS.items()}, **BASE}
    elif a.suite == "layer":
        confs = LAYER
    elif a.suite == "memory":
        confs = MEMORY
    elif a.suite == "fix":
        confs = FIX
    elif a.suite == "flip":
        confs = FLIP
    elif a.suite == "fmnist":
        confs = FMNIST
    elif a.suite == "modern_mnist":
        confs = MODERN_MNIST
    elif a.suite == "modern_fmnist":
        confs = MODERN_FMNIST
    elif a.suite == "round7":
        confs = ROUND7
    elif a.suite == "round8_fm":
        confs = ROUND8_FM
    elif a.suite == "round8_deep":
        confs = ROUND8_DEEP
    else:
        confs = SECANT
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
        if "--rho_bs" in cargs:  # larger ratio / curvature batch: charge its passes
            nb = int(cargs.split("--rho_bs")[1].split()[0])
            cost = (60000 / bs) * (1 + nb / bs if "--y_on_rho" in cargs else 2 + nb / (1.5 * bs))
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
