"""CIFAR-10 / ResNet-18 sweep: tuned baselines vs. ARCs-LSR1, same protocol as rounds 9-12 on F-MNIST.

  * Baselines (AdamW, SGD+momentum, Muon) run 2*E epochs, so they can be compared with ARCs-LSR1 (E epochs, about
    two gradients per step) both at the same epoch (E) and at the same compute (2E). The learning rate is chosen
    separately at epoch E and at epoch 2E, by the seed-0 TRAINING loss on the full training set; a value at the edge
    of the grid is extended by x3 or /3 (at most twice per direction).
  * ARCs-LSR1 (delta rule of the authors' code, double_grad, float32 buffers): m in {10, 50} x mu_min in
    {0.25, 1, 4}, and the diagonal B0 (p = 1) at m = 50 x mu_min in {0.25, 1}. A best mu_min at an edge is extended
    by x4 or /4. Selection by the seed-0 training loss at epoch E, separately with and without the diagonal B0.
  * Then seeds 1 and 2 for every selected configuration.

  python examples/sweep_cifar10.py --gpus 0,1,2,3            # one job per GPU at a time
  python examples/sweep_cifar10.py --gpus 0,1 --per_gpu 2    # two jobs per GPU (if memory allows)
  python examples/sweep_cifar10.py --dry_run                 # list the jobs of the first stage
  python examples/sweep_cifar10.py --smoke --gpus cpu --per_gpu 4   # whole pipeline on random data, on CPU
Skips runs whose JSON already exists, so it can be stopped and resumed. Results: results/cifar10/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
LRS = {"adamw": (3e-4, 1e-3, 3e-3, 1e-2), "sgd": (0.03, 0.1, 0.3, 1.0), "muon": (0.005, 0.015, 0.045, 0.135)}
ARCS_GRID = [(m, mu, 0.0) for m in (10, 50) for mu in (0.25, 1.0, 4.0)] + [(50, mu, 1.0) for mu in (0.25, 1.0)]
SEEDS = (1, 2)


class Sweep:
    def __init__(self, a):
        self.a = a
        self.out = a.out
        os.makedirs(os.path.join(self.out, "logs"), exist_ok=True)
        if a.smoke:
            self.common = ["--data", "fake", "--n_fake", "128", "--bs", "32"]  # 4 steps per epoch
            self.E = 1
            self.hist = {10: 3, 50: 5}  # small memory for the CPU smoke test
        else:
            self.common = ["--bs", str(a.bs)] + (["--augment"] if a.augment else [])
            self.E = a.epochs
            self.hist = {10: 10, 50: 50}
        self.data = "fake" if a.smoke else "cifar10"

    # ---------------------------------------------------------------- naming (must match train_cifar10.run_tag)
    def tag_base(self, opt, lr, seed):
        bs = "32" if self.a.smoke else str(self.a.bs)
        return f"{self.data}_{opt}_lr{lr:g}_bs{bs}" + ("_aug" if self.a.augment and not self.a.smoke else "") + f"_s{seed}"

    def tag_arcs(self, m, mu, p, seed):
        bs = "32" if self.a.smoke else str(self.a.bs)
        t = f"arcs_m{self.hist[m]}_mumin{mu:g}_code_dg" + (f"_diag{p:g}" if p else "")
        return f"{self.data}_{t}_bs{bs}" + ("_aug" if self.a.augment and not self.a.smoke else "") + f"_s{seed}"

    def job_base(self, opt, lr, seed):
        args = self.common + ["--opt", opt, "--lr", f"{lr:g}", "--epochs", str(2 * self.E), "--seed", str(seed)]
        return (2 * self.E, self.tag_base(opt, lr, seed), tuple(args))

    def job_arcs(self, m, mu, p, seed):
        args = self.common + ["--opt", "arcs", "--history", str(self.hist[m]), "--mu_min", f"{mu:g}",
                              "--delta_mode", "code", "--double_grad", "--epochs", str(self.E), "--seed", str(seed)]
        if p:
            args += ["--diag_power", f"{p:g}"]
        return (2 * self.E, self.tag_arcs(m, mu, p, seed), tuple(args))

    # ---------------------------------------------------------------- results
    def loss(self, tag, epoch):
        f = os.path.join(self.out, tag + ".json")
        if not os.path.exists(f):
            return None
        with open(f) as fh:
            ep = json.load(fh)["epochs"]
        if len(ep) <= epoch:
            return float("inf")  # stopped early (diverged)
        v = ep[epoch]["train_loss"]
        return v if v == v else float("inf")

    def tried(self, opt):
        out = {}
        for lr in self.lrs[opt]:
            a, b = self.loss(self.tag_base(opt, lr, 0), self.E), self.loss(self.tag_base(opt, lr, 0), 2 * self.E)
            if a is not None:
                out[lr] = (a, b)
        return out

    def arcs_tried(self):
        out = {}
        for m, mu, p in self.arcs_grid:
            v = self.loss(self.tag_arcs(m, mu, p, 0), self.E)
            if v is not None:
                out[(m, mu, p)] = v
        return out

    # ---------------------------------------------------------------- execution
    def run(self, jobs):
        jobs = [j for j in dict.fromkeys(jobs) if not os.path.exists(os.path.join(self.out, j[1] + ".json"))]
        jobs.sort(key=lambda j: -j[0])
        print(f"{len(jobs)} jobs, {sum(j[0] for j in jobs)} epoch-equivalents (one ARCs epoch ~ two baseline epochs)",
              flush=True)
        if self.a.dry_run:
            for j in jobs:
                print("  ", j[1])
            return
        q = queue.Queue()
        for j in jobs:
            q.put(j)
        slots = [g for g in self.a.gpus.split(",") for _ in range(self.a.per_gpu)]
        fails = []

        def worker(gpu):
            while True:
                try:
                    _, tag, args = q.get_nowait()
                except queue.Empty:
                    return
                env = dict(os.environ)
                if gpu == "cpu":
                    dev = ["--device", "cpu"]
                    env["OMP_NUM_THREADS"] = str(self.a.threads)
                else:
                    env["CUDA_VISIBLE_DEVICES"] = gpu
                    dev = ["--device", "cuda"]
                cmd = [PY, os.path.join(HERE, "train_cifar10.py")] + list(args) + dev + ["--out", self.out]
                with open(os.path.join(self.out, "logs", tag + ".log"), "w") as f:
                    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
                print(f"done {tag} rc={rc} (gpu {gpu})", flush=True)
                if rc:
                    fails.append(tag)

        ts = [threading.Thread(target=worker, args=(g,)) for g in slots]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        print(f"failed: {len(fails)} {fails}", flush=True)

    def main(self):
        self.lrs = {k: list(v) for k, v in LRS.items()}
        self.arcs_grid = list(ARCS_GRID)
        # stage 1: seed-0 grids
        self.run([self.job_base(o, lr, 0) for o, lrs in self.lrs.items() for lr in lrs] +
                 [self.job_arcs(m, mu, p, 0) for m, mu, p in self.arcs_grid])
        if self.a.dry_run:
            return
        # stage 2: extend edges
        ext = {}
        while True:
            jobs = []
            for o in self.lrs:
                tl = self.tried(o)
                lrs = sorted(tl)
                for i in (0, 1):
                    best = min(lrs, key=lambda lr: tl[lr][i])
                    for d, new in (("down", best / 3), ("up", best * 3)):
                        edge = best == (lrs[0] if d == "down" else lrs[-1])
                        if edge and ext.get((o, d), 0) < 2 and new not in self.lrs[o]:
                            ext[(o, d)] = ext.get((o, d), 0) + 1
                            self.lrs[o].append(new)
                            jobs.append(self.job_base(o, new, 0))
            g = self.arcs_tried()
            for p in (0.0, 1.0):
                fam = {k: v for k, v in g.items() if k[2] == p}
                if not fam:
                    continue
                m, mu, _ = min(fam, key=fam.get)
                mus = sorted(k[1] for k in fam if k[0] == m)
                for d, new in (("down", mu / 4), ("up", mu * 4)):
                    edge = mu == (mus[0] if d == "down" else mus[-1])
                    if edge and ext.get(("arcs", p, d), 0) < 2 and (m, new, p) not in self.arcs_grid:
                        ext[("arcs", p, d)] = ext.get(("arcs", p, d), 0) + 1
                        self.arcs_grid.append((m, new, p))
                        jobs.append(self.job_arcs(m, new, p, 0))
            if not jobs:
                break
            print("extending:", [j[1] for j in jobs], flush=True)
            self.run(jobs)
        # stage 3: selection and seeds 1-2
        sel, jobs = {"E": self.E}, []
        for o in self.lrs:
            tl = self.tried(o)
            b1 = min(tl, key=lambda lr: tl[lr][0])
            b2 = min(tl, key=lambda lr: tl[lr][1])
            lrs = sorted(tl)
            sel[o] = {"lr_epochE": b1, "lr_epoch2E": b2, "at_edge": any(b in (lrs[0], lrs[-1]) for b in (b1, b2)),
                      "tried": {f"{lr:g}": {"epochE": v[0], "epoch2E": v[1]} for lr, v in sorted(tl.items())}}
            for lr in {b1, b2}:
                jobs += [self.job_base(o, lr, s) for s in SEEDS]
        g = self.arcs_tried()
        for p in (0.0, 1.0):
            fam = {k: v for k, v in g.items() if k[2] == p}
            if fam:
                m, mu, _ = min(fam, key=fam.get)
                mus = sorted(k[1] for k in fam if k[0] == m)
                sel[f"arcs_diag{p:g}"] = {"m": m, "mu_min": mu, "diag_power": p, "at_edge": mu in (mus[0], mus[-1]),
                                          "tried": {f"m={k[0]},mu_min={k[1]:g}": v for k, v in sorted(fam.items())}}
                jobs += [self.job_arcs(m, mu, p, s) for s in SEEDS]
        with open(os.path.join(self.out, "selection.json"), "w") as f:
            json.dump(sel, f, indent=1)
        print("selection:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "tried"} if isinstance(v, dict)
                                        else v for k, v in sel.items()}), flush=True)
        self.run(jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0", help="comma list of GPU ids, or 'cpu'")
    ap.add_argument("--per_gpu", type=int, default=1, help="concurrent jobs per GPU")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads per job when --gpus cpu")
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=20, help="E: ARCs-LSR1 epochs (baselines run 2E)")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="random data, tiny runs: tests the whole pipeline")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or os.path.join(ROOT, "results", "cifar10_smoke" if a.smoke else "cifar10")
    Sweep(a).main()


if __name__ == "__main__":
    main()
