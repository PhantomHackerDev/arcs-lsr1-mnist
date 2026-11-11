"""Run one MNIST training experiment and dump a JSON log.

Network: the paper's MNIST classifier, 784-500-ReLU-10 (397,510 params).
Dropout is omitted on purpose: with dropout the ARC ratio test
rho = (f(x) - f(x+s)) / pred compares two different random sub-networks,
which would mix optimizer noise with dropout noise.

Example
  python src/run.py --opt arcs --bs 256 --delta_mode safe --y_mode cross
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arcs_lsr1 import ARCsLSR1  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHS = {
    "paper": [784, 500, 10],  # the paper's MNIST classifier, 397,510 params
    "deep": [784, 512, 256, 128, 10],  # 4 linear layers, 567,434 params
}
WIDTHS = ARCHS["paper"]
SHAPES, LAYER_BLOCKS, N_PARAMS = [], [], 0


def set_arch(name):
    global WIDTHS, SHAPES, LAYER_BLOCKS, N_PARAMS
    WIDTHS = ARCHS[name]
    SHAPES, LAYER_BLOCKS, i = [], [], 0
    for a, b in zip(WIDTHS[:-1], WIDTHS[1:]):
        SHAPES += [(b, a), (b,)]
        LAYER_BLOCKS.append((i, i + a * b + b))
        i += a * b + b
    N_PARAMS = i


set_arch("paper")


def load_mnist():
    cache = os.path.join(ROOT, "data", "mnist_tensors.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    from torchvision import datasets

    tr = datasets.MNIST(os.path.join(ROOT, "data"), train=True, download=True)
    te = datasets.MNIST(os.path.join(ROOT, "data"), train=False, download=True)

    def prep(ds):
        x = ds.data.float().div(255.0).sub(0.1307).div(0.3081).reshape(-1, 784)
        return x.contiguous(), ds.targets.clone()

    out = (*prep(tr), *prep(te))
    torch.save(out, cache)
    return out


def init_params(seed):
    torch.manual_seed(seed)
    layers = [nn.Linear(a, b) for a, b in zip(WIDTHS[:-1], WIDTHS[1:])]
    return torch.cat([p.detach().reshape(-1) for l in layers for p in (l.weight, l.bias)]).contiguous()


def unflat(x):
    out, i = [], 0
    for sh in SHAPES:
        n = math.prod(sh)
        out.append(x[i:i + n].view(sh))
        i += n
    return out


def forward(x, X):
    ps = unflat(x)
    h = X
    nl = len(ps) // 2
    for i in range(nl):
        h = F.linear(h, ps[2 * i], ps[2 * i + 1])
        if i < nl - 1:
            h = F.relu(h)
    return h


def loss_fn(x, X, T):
    return F.cross_entropy(forward(x, X), T)


def make_closures(Xb, Tb):
    def loss_and_grad(x):
        xv = x.detach().requires_grad_(True)
        L = loss_fn(xv, Xb, Tb)
        (g,) = torch.autograd.grad(L, xv)
        return float(L.detach()), g.detach()

    @torch.no_grad()
    def loss_only(x):
        return float(loss_fn(x, Xb, Tb))

    return loss_and_grad, loss_only


@torch.no_grad()
def evaluate(x, X, T, chunk=10000):
    tot_loss, correct = 0.0, 0
    for i in range(0, X.shape[0], chunk):
        out = forward(x, X[i:i + chunk])
        tot_loss += float(F.cross_entropy(out, T[i:i + chunk], reduction="sum"))
        correct += int((out.argmax(1) == T[i:i + chunk]).sum())
    return tot_loss / X.shape[0], correct / X.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", default="arcs", choices=["arcs", "sgd", "adam"])
    ap.add_argument("--arch", default="paper", choices=list(ARCHS))
    ap.add_argument("--mu_per_layer", action="store_true")
    ap.add_argument("--hess_beta", type=float, default=0.0)
    ap.add_argument("--rank_cap", type=int, default=0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--history", type=int, default=10)
    ap.add_argument("--mu0", type=float, default=1.0)
    ap.add_argument("--delta_mode", default="safe")
    ap.add_argument("--delta_fallback", type=float, default=1.0)
    ap.add_argument("--y_mode", default="cross")
    ap.add_argument("--per_layer", action="store_true")
    ap.add_argument("--grad_beta", type=float, default=0.0)
    ap.add_argument("--unstable_cubic", action="store_true")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--probe_every", type=int, default=50)
    ap.add_argument("--probe_size", type=int, default=4096)
    ap.add_argument("--max_iters", type=int, default=0, help="stop after this many iterations (0 = no cap)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    set_arch(a.arch)
    Xtr, Ttr, Xte, Tte = load_mnist()
    x = init_params(a.seed)
    x0 = x.clone()

    def layer_move(xc):
        """relative distance each layer has moved from its initialization"""
        return [float((xc[s:e] - x0[s:e]).norm() / x0[s:e].norm()) for s, e in LAYER_BLOCKS]
    gen = torch.Generator().manual_seed(1000 + a.seed)
    probe_idx = torch.randperm(Xtr.shape[0], generator=torch.Generator().manual_seed(12345))[: a.probe_size]
    Xp, Tp = Xtr[probe_idx], Ttr[probe_idx]

    def hvp(v64):
        xv = x.detach().requires_grad_(True)
        L = loss_fn(xv, Xp, Tp)
        (g,) = torch.autograd.grad(L, xv, create_graph=True)
        (hv,) = torch.autograd.grad(g @ v64.to(g.dtype), xv)
        return hv.detach().to(torch.float64)

    name = a.tag or (
        f"{a.opt}_bs{a.bs}" + (f"_{a.delta_mode}_{a.y_mode}" + ("_layer" if a.per_layer else "")
                               + (f"_gb{a.grad_beta}" if a.grad_beta > 0 else "") if a.opt == "arcs" else "")
        + f"_s{a.seed}")
    os.makedirs(a.outdir, exist_ok=True)

    opt = None
    if a.opt == "arcs":
        opt = ARCsLSR1(
            N_PARAMS, history=a.history, mu0=a.mu0, delta_mode=a.delta_mode,
            delta_fallback=a.delta_fallback, y_mode=a.y_mode,
            blocks=LAYER_BLOCKS if a.per_layer else None, grad_beta=a.grad_beta,
            stable_cubic=not a.unstable_cubic, mu_per_layer=a.mu_per_layer,
            hess_beta=a.hess_beta, rank_cap=a.rank_cap)
    else:
        xp = x.clone().requires_grad_(True)
        if a.opt == "sgd":
            topt = torch.optim.SGD([xp], lr=a.lr or 0.1, momentum=0.9)
        else:
            topt = torch.optim.Adam([xp], lr=a.lr or 1e-3, eps=1e-6)

    log = {"args": vars(a), "name": name, "iters": [], "probes": [], "epochs": []}
    tl, ta = evaluate(x, Xtr, Ttr)
    vl, va = evaluate(x, Xte, Tte)
    log["epochs"].append({"epoch": 0, "train_loss": tl, "train_acc": ta, "test_loss": vl, "test_acc": va,
                          "time": 0.0, "n_grad": 0, "iter": 0})
    t0 = time.time()
    n_grad, it = 0, 0
    stop = False
    for ep in range(1, a.epochs + 1):
        perm = torch.randperm(Xtr.shape[0], generator=gen)
        for bi in range(0, Xtr.shape[0] - a.bs + 1, a.bs):
            idx = perm[bi:bi + a.bs]
            Xb, Tb = Xtr[idx], Ttr[idx]
            it += 1
            if opt is not None:
                lg, lo = make_closures(Xb, Tb)
                d = opt.step(x, lg, lo)
                n_grad += d["n_grad"]
                d["it"] = it
                log["iters"].append(d)
                if a.probe_every and it % a.probe_every == 0 and torch.isfinite(opt._last_s).all():
                    p = opt.curvature_probe(hvp, opt._last_s)
                    p["it"] = it
                    p["mu"] = d["mu"]
                    log["probes"].append(p)
            else:
                topt.zero_grad()
                L = loss_fn(xp, Xb, Tb)
                L.backward()
                topt.step()
                n_grad += 1
                if it % 10 == 0:
                    log["iters"].append({"it": it, "loss": float(L)})
            if a.max_iters and it >= a.max_iters:
                stop = True
                break
        if opt is None:
            x = xp.detach()
        tl, ta = evaluate(x, Xtr, Ttr)
        vl, va = evaluate(x, Xte, Tte)
        rec = {"epoch": ep, "train_loss": tl, "train_acc": ta, "test_loss": vl, "test_acc": va,
               "time": time.time() - t0, "n_grad": n_grad, "iter": it, "move_b": layer_move(x)}
        if opt is not None:
            rec["mu"] = opt.mu
            rec["mu_b"] = list(opt.mus)
            rec["n_skip"] = [b.n_skip for b in opt.blocks]
            rec["n_add"] = [b.n_add for b in opt.blocks]
        log["epochs"].append(rec)
        print(f"[{name}] ep {ep} train_loss {tl:.4f} test_acc {va:.4f} t {rec['time']:.0f}s"
              + (f" mu {opt.mu:.3g}" if opt else ""), flush=True)
        if stop or not math.isfinite(tl):
            break

    with open(os.path.join(a.outdir, name + ".json"), "w") as f:
        json.dump(log, f)


if __name__ == "__main__":
    main()
