"""Pre-flight checks for the GPU experiments (run on the machine that will run the sweep).

  python examples/test_examples.py --device cuda
1. ModuleMuon (nn.Module version) == FlatMuon (the flat version used on MNIST / F-MNIST), 5 steps on the F-MNIST CNN.
2. ARCs-LSR1 through ARCsLSR1Optim gives the same steps on the device as on the CPU (float64 state, small MLP).
3. train_cifar10.py runs 3 steps with every optimizer on the device (random data) and logs finite losses.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from arcs_optim import ARCsLSR1Optim  # noqa: E402
from baselines import FlatMuon  # noqa: E402
from optim_baselines import ModuleMuon  # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}", flush=True)


def test_muon(dev):
    torch.manual_seed(0)
    convs = [nn.Conv2d(1, 10, 5), nn.Conv2d(10, 20, 5)]
    fc1, fc2 = nn.Linear(320, 50), nn.Linear(50, 10)
    layers = convs + [fc1, fc2]
    model = nn.ModuleList(layers).to(dev)
    shapes = [tuple(p.shape) for l in layers for p in (l.weight, l.bias)]
    xp = torch.cat([p.detach().reshape(-1) for l in layers for p in (l.weight, l.bias)]).clone().requires_grad_(True)

    def fwd_mod(X):
        h = F.relu(F.max_pool2d(convs[0](X), 2))
        h = F.relu(F.max_pool2d(convs[1](h), 2))
        return fc2(F.relu(fc1(h.flatten(1))))

    def fwd_flat(x, X):
        ps, i = [], 0
        for sh in shapes:
            n = int(torch.tensor(sh).prod())
            ps.append(x[i:i + n].view(sh))
            i += n
        h = F.relu(F.max_pool2d(F.conv2d(X, ps[0], ps[1]), 2))
        h = F.relu(F.max_pool2d(F.conv2d(h, ps[2], ps[3]), 2))
        return F.linear(F.relu(F.linear(h.flatten(1), ps[4], ps[5])), ps[6], ps[7])

    o1 = ModuleMuon.for_model(model, fc2, lr=0.02, wd=0.01)
    o2 = FlatMuon(xp, shapes, lr=0.02, wd=0.01)
    g = torch.Generator().manual_seed(1)
    for _ in range(5):
        X, T = torch.randn(32, 1, 28, 28, generator=g).to(dev), torch.randint(0, 10, (32,), generator=g).to(dev)
        o1.zero_grad()
        F.cross_entropy(fwd_mod(X), T).backward()
        o1.step()
        o2.zero_grad()
        F.cross_entropy(fwd_flat(xp, X), T).backward()
        o2.step()
    flat1 = torch.cat([p.detach().reshape(-1) for l in layers for p in (l.weight, l.bias)])
    err = float((flat1 - xp.detach()).abs().max())
    check("ModuleMuon == FlatMuon (5 steps, F-MNIST CNN)", err < 1e-5, f"(max abs diff {err:.1e})")


def test_arcs_device(dev, force=False):
    def run(d):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(20, 32), nn.Tanh(), nn.Linear(32, 5)).to(d)
        opt = ARCsLSR1Optim(model.parameters(), y_mode="same", history=4, delta_mode="code", double_grad=True)
        g = torch.Generator().manual_seed(2)
        rhos = []
        for _ in range(15):
            X, T = torch.randn(64, 20, generator=g).to(d), torch.randint(0, 5, (64,), generator=g).to(d)

            def closure(backward=True):
                loss = F.cross_entropy(model(X), T)
                if backward:
                    loss.backward()
                return loss
            opt.step(closure)
            rhos.append(opt.last_info["rho"])
        return torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()]), rhos

    if dev.type == "cpu" and not force:
        print("[SKIP] ARCs-LSR1 on the device == on the CPU (device is the CPU)")
        return
    (x_c, r_c), (x_d, r_d) = run(torch.device("cpu")), run(dev)
    rel = float((x_c - x_d).norm() / x_c.norm())
    check(f"ARCs-LSR1 on {dev} == on the CPU (15 steps)", rel < 1e-4, f"(rel err {rel:.1e}, accepted "
          f"{sum(r >= 0.1 for r in r_d)}/15)")


def test_train_script(dev):
    import train_cifar10

    with tempfile.TemporaryDirectory() as out:
        for opt in (["--opt", "arcs", "--double_grad", "--history", "5"],
                    ["--opt", "arcs", "--diag_power", "1", "--history", "5", "--augment"],
                    ["--opt", "muon"], ["--opt", "adamw"], ["--opt", "sgd", "--lr", "0.01"]):
            log = train_cifar10.main(["--data", "fake", "--n_fake", "96", "--bs", "32", "--epochs", "1",
                                      "--device", str(dev), "--out", out] + opt)
            e = log["epochs"][-1]
            check(f"train_cifar10.py {' '.join(opt)} on {dev}", e["steps"] == 3 and e["train_loss"] == e["train_loss"],
                  f"(train loss {e['train_loss']:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                       ("cpu" if a.device == "auto" else a.device))
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else ""), flush=True)
    test_muon(dev)
    test_arcs_device(dev)
    test_train_script(dev)
    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
