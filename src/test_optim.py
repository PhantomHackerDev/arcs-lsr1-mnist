"""The torch.optim wrapper must reproduce the functional implementation used in
all experiments (run: python src/test_optim.py)."""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run  # noqa: E402
from arcs_lsr1 import ARCsLSR1  # noqa: E402
from arcs_optim import ARCsLSR1Optim  # noqa: E402

torch.set_num_threads(1)
run.set_arch("deep")
Xtr, Ttr, _, _ = run.load_mnist()
gen = torch.Generator().manual_seed(7)
batches = [torch.randint(0, Xtr.shape[0], (128,), generator=gen) for _ in range(30)]
KW = dict(history=5, y_mode="same", mu_per_layer=True, mu_floor_kappa=1.0)
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


def functional():
    x = run.init_params(0)
    opt = ARCsLSR1(run.N_PARAMS, blocks=run.LAYER_BLOCKS, **KW)
    losses = []
    for idx in batches:
        lg, lo = run.make_closures(Xtr[idx], Ttr[idx])
        losses.append(opt.step(x, lg, lo)["loss"])
    return x, losses


def wrapped(with_kw: bool):
    x0 = run.init_params(0)
    layers = [nn.Linear(a, b) for a, b in zip(run.WIDTHS[:-1], run.WIDTHS[1:])]
    with torch.no_grad():
        i = 0
        for l in layers:
            for p in (l.weight, l.bias):
                p.copy_(x0[i:i + p.numel()].view_as(p))
                i += p.numel()
    mods = []
    for l in layers[:-1]:
        mods += [l, nn.ReLU()]
    model = nn.Sequential(*mods, layers[-1])
    opt = ARCsLSR1Optim([{"params": l.parameters()} for l in layers], per_layer=True, **KW)
    losses = []
    for idx in batches:
        Xb, Tb = Xtr[idx], Ttr[idx]
        if with_kw:
            def closure(backward=True):
                loss = F.cross_entropy(model(Xb), Tb)
                if backward:
                    loss.backward()
                return loss
        else:
            def closure():
                opt.zero_grad()
                loss = F.cross_entropy(model(Xb), Tb)
                loss.backward()
                return loss
        losses.append(float(opt.step(closure)))
    x = torch.cat([p.detach().reshape(-1) for l in layers for p in (l.weight, l.bias)])
    return x, losses


xf, lf = functional()
for kw in (True, False):
    xw, lw = wrapped(kw)
    err = float((xw - xf).norm() / xf.norm())
    lerr = max(abs(a - b) for a, b in zip(lf, lw))
    label = "closure(backward=...)" if kw else "plain closure()"
    check(f"wrapper == functional, {label}: final parameters", err < 1e-5, f"(rel diff {err:.1e})")
    check(f"wrapper == functional, {label}: loss trajectory", lerr < 1e-5, f"(max diff {lerr:.1e})")
check("training made progress", lf[-1] < 0.5 * lf[0], f"({lf[0]:.3f} -> {lf[-1]:.3f})")
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
