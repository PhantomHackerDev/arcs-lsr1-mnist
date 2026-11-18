"""CIFAR-10 / ResNet-18 with ARCs-LSR1 (torch.optim interface) or a baseline, on CPU or GPU.

A template for the larger experiments: a standard training loop, the model and data on --device, and
ARCs-LSR1 through ARCsLSR1Optim with a closure that evaluates the same mini-batch on every call.

  # GPU, CIFAR-10 (downloads to ./data on first use)
  python examples/train_cifar10.py --device cuda --opt arcs --bs 1024 --history 50 --vec_dtype float32 --double_grad
  python examples/train_cifar10.py --device cuda --opt adamw --lr 3e-3 --bs 1024
  # quick CPU smoke test with random data
  python examples/train_cifar10.py --data fake --device cpu --opt arcs --bs 32 --max_steps 3

Notes
  * BatchNorm: every closure call in train mode normalises with the statistics of the same batch, so f(x) and
    f(x + s) in the ratio test are evaluated on the same function; the running statistics are updated on each
    call (2-3 times per ARCs step), which only affects evaluation mode.
  * Data augmentation is off by default (--augment turns it on), so the training loss measures the optimizer.
  * The L-SR1 buffer holds 2m x n values: about 4.5 GB for ResNet-18 at m = 50 in float32.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from arcs_optim import ARCsLSR1Optim  # noqa: E402

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def resnet18_cifar(num_classes=10):
    import torchvision

    m = torchvision.models.resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)  # CIFAR stem: 3x3 conv, no max-pool
    m.maxpool = nn.Identity()
    return m


def load(name, augment, n_fake=512):
    import torchvision
    import torchvision.transforms as T

    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if name == "fake":
        mk = lambda n, s: torchvision.datasets.FakeData(n, (3, 32, 32), 10, T.Compose(norm), random_offset=s)  # noqa: E731
        return mk(n_fake, 0), mk(n_fake // 4, 1)
    aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if augment else []
    root = os.path.join(ROOT, "data")
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=T.Compose(aug + norm))
    te = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=T.Compose(norm))
    return tr, te


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    tot, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
        out = model(xb)
        tot += float(F.cross_entropy(out, yb, reduction="sum"))
        correct += int((out.argmax(1) == yb).sum())
        n += yb.numel()
    model.train()
    return tot / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cifar10", choices=["cifar10", "fake"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--opt", default="arcs", choices=["arcs", "adamw", "sgd"])
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None, help="baselines")
    ap.add_argument("--wd", type=float, default=0.01, help="AdamW weight decay")
    # ARCs-LSR1 (defaults: the best F-MNIST settings of rounds 9-11)
    ap.add_argument("--history", type=int, default=50)
    ap.add_argument("--mu_min", type=float, default=1.0)
    ap.add_argument("--eta2", type=float, default=0.5)
    ap.add_argument("--delta_mode", default="safe_lm")
    ap.add_argument("--safe_frac", type=float, default=0.5)
    ap.add_argument("--double_grad", action="store_true")
    ap.add_argument("--vec_dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cifar10"))
    a = ap.parse_args()

    dev = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                       ("cpu" if a.device == "auto" else a.device))
    torch.manual_seed(a.seed)
    tr, te = load(a.data, a.augment)
    g = torch.Generator().manual_seed(1000 + a.seed)
    kw = dict(num_workers=a.workers, pin_memory=dev.type == "cuda")
    train_loader = torch.utils.data.DataLoader(tr, a.bs, shuffle=True, drop_last=True, generator=g, **kw)
    test_loader = torch.utils.data.DataLoader(te, 1000, **kw)
    model = resnet18_cifar().to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    if a.opt == "arcs":
        opt = ARCsLSR1Optim(model.parameters(), y_mode="same", history=a.history, mu_min=a.mu_min, eta2=a.eta2,
                            delta_mode=a.delta_mode, safe_frac=a.safe_frac, double_grad=a.double_grad,
                            vec_dtype=getattr(torch, a.vec_dtype))
    elif a.opt == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr or 1e-3, weight_decay=a.wd)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=a.lr or 0.1, momentum=0.9)
    tag = f"{a.data}_{a.opt}_bs{a.bs}" + (f"_m{a.history}_{a.delta_mode}" + ("_dg" if a.double_grad else "")
                                          if a.opt == "arcs" else f"_lr{a.lr}") + f"_s{a.seed}"
    log = {"args": vars(a), "device": str(dev), "n_params": n_params, "epochs": [], "steps": []}
    print(f"[{tag}] {n_params:,} parameters on {dev}", flush=True)

    t0, step, samples_fb = time.time(), 0, 0
    model.train()
    for ep in range(1, a.epochs + 1):
        run_loss, nb = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            if a.opt == "arcs":
                def closure(backward=True):
                    loss = F.cross_entropy(model(xb), yb)
                    if backward:
                        loss.backward()
                    return loss
                loss = float(opt.step(closure))
                info = opt.last_info
                samples_fb += info["n_grad"] * a.bs
                log["steps"].append({k: info[k] for k in ("loss", "rho", "acc", "mu", "s_norm", "n_grad")}
                                    | {"delta": info["delta"][0], "lm": info["lm"][0]})
            else:
                opt.zero_grad(set_to_none=True)
                lt = F.cross_entropy(model(xb), yb)
                lt.backward()
                opt.step()
                loss = float(lt.detach())
                samples_fb += a.bs
            run_loss, nb, step = run_loss + loss, nb + 1, step + 1
            if not math.isfinite(loss) or (a.max_steps and step >= a.max_steps):
                break
        te_loss, te_acc = evaluate(model, test_loader, dev)
        rec = {"epoch": ep, "train_loss_running": run_loss / max(nb, 1), "test_loss": te_loss, "test_acc": te_acc,
               "time": time.time() - t0, "steps": step, "grad_epochs": samples_fb / len(tr)}
        if dev.type == "cuda":
            rec["max_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 1e9
        log["epochs"].append(rec)
        print(f"[{tag}] ep {ep} train(running) {rec['train_loss_running']:.4f} test_acc {te_acc:.4f} "
              f"t {rec['time']:.0f}s", flush=True)
        if not math.isfinite(loss) or (a.max_steps and step >= a.max_steps):
            break
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, tag + ".json"), "w") as f:
        json.dump(log, f)


if __name__ == "__main__":
    main()
