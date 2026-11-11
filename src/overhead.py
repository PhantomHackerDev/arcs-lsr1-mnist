"""Per-iteration / per-epoch computational overhead, measured on an otherwise idle machine.

For each optimizer: gradient evaluations (forward+backward), extra forward-only
passes, model FLOPs (FlopCounterMode, per sample), optimizer FLOPs (dominant
O(n*m) terms), and wall time split into model evaluations vs. optimizer algebra.

  python src/overhead.py --arch deep --bs 256 --iters 150 --threads 8
  -> results/overhead.md
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run  # noqa: E402
from arcs_lsr1 import ARCsLSR1  # noqa: E402

CONFIGS = [
    ("SGD + momentum", "sgd", {}),
    ("Adam", "adam", {}),
    ("ARCs-LSR1, full network, m=10", "arcs", dict(history=10)),
    ("per-layer models, shared μ, m=10", "arcs", dict(history=10, per_layer=True)),
    ("per-layer μ_l + floor, m=10", "arcs", dict(history=10, per_layer=True, mu_per_layer=True, mu_floor_kappa=1.0)),
    ("per-layer μ_l, m=2", "arcs", dict(history=2, per_layer=True, mu_per_layer=True)),
    ("per-layer μ_l, m=2, Hessian momentum", "arcs",
     dict(history=2, per_layer=True, mu_per_layer=True, hess_beta=0.1, rank_cap=2)),
]


def bench(kind, kw, a, Xtr, Ttr):
    torch.manual_seed(0)
    x = run.init_params(0)
    gen = torch.Generator().manual_seed(1)
    idxs = [torch.randint(0, Xtr.shape[0], (a.bs,), generator=gen) for _ in range(a.iters + a.warmup)]
    t_model = 0.0
    n_grad = n_fwd = 0

    def timed(f):
        def g(*args):
            nonlocal t_model
            t0 = time.perf_counter()
            out = f(*args)
            t_model += time.perf_counter() - t0
            return out
        return g

    if kind == "arcs":
        kw = dict(kw)
        per_layer = kw.pop("per_layer", False)
        opt = ARCsLSR1(run.N_PARAMS, y_mode="same", blocks=run.LAYER_BLOCKS if per_layer else None, **kw)
    else:
        xp = x.clone().requires_grad_(True)
        topt = torch.optim.SGD([xp], lr=0.1, momentum=0.9) if kind == "sgd" else torch.optim.Adam([xp], lr=1e-3)
    flops0 = 0
    for k, idx in enumerate(idxs):
        if k == a.warmup:  # start measuring
            t_model, n_grad, n_fwd = 0.0, 0, 0
            t_start = time.perf_counter()
            flops0 = opt.flops if kind == "arcs" else 0
        Xb, Tb = Xtr[idx], Ttr[idx]
        if kind == "arcs":
            lg, lo = run.make_closures(Xb, Tb)
            d = opt.step(x, timed(lg), timed(lo))
            n_grad += d["n_grad"]
            n_fwd += d["n_fwd"]
        else:
            t0 = time.perf_counter()
            topt.zero_grad()
            L = run.loss_fn(xp, Xb, Tb)
            L.backward()
            t_model += time.perf_counter() - t0
            topt.step()
            n_grad += 1
    t_total = time.perf_counter() - t_start
    it = a.iters
    return {
        "grad_per_it": n_grad / it, "fwd_per_it": n_fwd / it,
        "ms_total": 1e3 * t_total / it, "ms_model": 1e3 * t_model / it,
        "ms_opt": 1e3 * (t_total - t_model) / it,
        "opt_mflops_per_it": ((opt.flops - flops0) / it / 1e6) if kind == "arcs" else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="deep")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run.set_arch(a.arch)
    Xtr, Ttr, _, _ = run.load_mnist()
    f_fwd, f_fb = run.model_flops_per_sample()
    it_per_epoch = Xtr.shape[0] // a.bs
    rows = []
    for name, kind, kw in CONFIGS:
        r = bench(kind, kw, a, Xtr, Ttr)
        r["model_gflops_per_epoch"] = it_per_epoch * a.bs * (r["grad_per_it"] * f_fb + r["fwd_per_it"] * f_fwd) / 1e9
        r["opt_gflops_per_epoch"] = it_per_epoch * r["opt_mflops_per_it"] / 1e3
        r["s_per_epoch"] = it_per_epoch * r["ms_total"] / 1e3
        rows.append((name, r))
        print(name, {k: round(v, 3) for k, v in r.items()}, flush=True)
    base = rows[0][1]["s_per_epoch"]
    lines = [f"### Overhead per iteration and per epoch ({a.arch} net, {run.N_PARAMS:,} params, batch {a.bs}, "
             f"{it_per_epoch} iterations/epoch, {a.threads} CPU threads, idle machine, mean over {a.iters} iterations)",
             "",
             "| optimizer | grad evals / it | extra forwards / it | model GFLOPs / epoch | optimizer GFLOPs / epoch "
             "| ms / it: model | ms / it: optimizer | s / epoch | × SGD time |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in rows:
        og = "–" if r["opt_gflops_per_epoch"] != r["opt_gflops_per_epoch"] else f"{r['opt_gflops_per_epoch']:.1f}"
        lines.append(f"| {name} | {r['grad_per_it']:.2f} | {r['fwd_per_it']:.2f} | {r['model_gflops_per_epoch']:.0f} | "
                     f"{og} | {r['ms_model']:.1f} | {r['ms_opt']:.1f} | {r['s_per_epoch']:.1f} | "
                     f"{r['s_per_epoch'] / base:.1f} |")
    lines += ["", f"Model FLOPs per sample (torch FlopCounterMode): forward {f_fwd:.3g}, forward+backward {f_fb:.3g}. "
              "Optimizer FLOPs count only the products with parameter-sized arrays (O(n·m)); O(n) vector "
              "operations are not counted, and for SGD/Adam the optimizer cost is a few O(n) operations per step."]
    out = os.path.join(run.ROOT, "results", "overhead.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
