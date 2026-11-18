"""CIFAR-10 / ResNet-18 with ARCs-LSR1 (torch.optim interface) or a baseline, on CPU or GPU.

The whole dataset is kept on the device as uint8 (150 MB) and batched there, so there is no DataLoader and the
batch order depends only on the seed. Each epoch logs the loss on the FULL training set (eval mode, the quantity the
optimizers are compared on), the test loss / accuracy, the compute used so far and the wall time.

  python examples/train_cifar10.py --device cuda --opt arcs --bs 1024 --epochs 20 --history 50 --double_grad
  python examples/train_cifar10.py --device cuda --opt muon --lr 0.015 --bs 1024 --epochs 40
  python examples/train_cifar10.py --data fake --device cpu --opt arcs --bs 32 --max_steps 3      # smoke test

Notes
  * ARCs-LSR1 defaults: same-batch pairs, the delta rule of the authors' code ('code'), float32 L-SR1 buffers.
    Each step evaluates the batch 2-3 times (gradient, gradient at the trial point, and the ratio test reuses it),
    so one ARCs epoch costs about two baseline epochs; "grad_epochs" in the log counts this.
  * BatchNorm: every closure call in train mode normalises with the statistics of the same batch, so f(x) and
    f(x + s) in the ratio test are the same function; the running statistics (used in eval mode) are updated on each
    call, i.e. 2-3 times per ARCs step.
  * Augmentation (random crop + flip, on the device) is off by default, so the training loss measures the optimizer;
    the augmented batch is drawn once per step, before the closure.
  * Memory: the L-SR1 buffer holds 2m x n values, about 4.5 GB for ResNet-18 (11.2M parameters) at m = 50 in float32.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from arcs_optim import ARCsLSR1Optim  # noqa: E402
from optim_baselines import ModuleMuon  # noqa: E402

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def resnet18_cifar(num_classes=10):
    import torchvision

    m = torchvision.models.resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)  # CIFAR stem: 3x3 conv, no max-pool
    m.maxpool = nn.Identity()
    return m


def load_data(name, dev, n_fake=512):
    """(Xtr, Ttr, Xte, Tte) on dev; X as uint8 (N, 3, 32, 32)."""
    if name == "fake":
        g = torch.Generator().manual_seed(0)
        mk = lambda n: (torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8, generator=g),  # noqa: E731
                        torch.randint(0, 10, (n,), generator=g))
        (a, b), (c, d) = mk(n_fake), mk(n_fake // 4)
    else:
        import torchvision

        root = os.path.join(ROOT, "data")
        tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True)
        a, b = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(tr.targets)
        c, d = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(te.targets)
    return a.to(dev), b.to(dev), c.to(dev), d.to(dev)


def prep(xu8, mean, std):
    return (xu8.float().div_(255.0) - mean) / std


def augment(x, gen):
    """random crop (zero padding 4) + horizontal flip, per sample, on x's device"""
    B = x.shape[0]
    p = F.pad(x, (4, 4, 4, 4)).permute(0, 2, 3, 1)  # B, 40, 40, C
    oy = torch.randint(0, 9, (B,), generator=gen).to(x.device)
    ox = torch.randint(0, 9, (B,), generator=gen).to(x.device)
    base = torch.arange(32, device=x.device)
    rows, cols = (oy[:, None] + base)[:, :, None], (ox[:, None] + base)[:, None, :]
    out = p[torch.arange(B, device=x.device)[:, None, None], rows, cols].permute(0, 3, 1, 2)
    flip = (torch.rand(B, generator=gen) < 0.5).to(x.device)[:, None, None, None]
    return torch.where(flip, out.flip(3), out)


@torch.no_grad()
def evaluate(model, X, T, mean, std, chunk=2000):
    was = model.training
    model.eval()
    tot, correct = 0.0, 0
    for i in range(0, X.shape[0], chunk):
        out = model(prep(X[i:i + chunk], mean, std))
        tot += float(F.cross_entropy(out, T[i:i + chunk], reduction="sum"))
        correct += int((out.argmax(1) == T[i:i + chunk]).sum())
    model.train(was)
    return tot / X.shape[0], correct / X.shape[0]


def run_tag(a):
    if a.opt == "arcs":
        t = f"arcs_m{a.history}_mumin{a.mu_min:g}_{a.delta_mode}" + ("_dg" if a.double_grad else "")
        t += f"_diag{a.diag_power:g}" if a.diag_power else ""
    else:
        t = f"{a.opt}_lr{a.lr:g}"
    return f"{a.data}_{t}_bs{a.bs}" + ("_aug" if a.augment else "") + f"_s{a.seed}"


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cifar10", choices=["cifar10", "fake"])
    ap.add_argument("--n_fake", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--opt", default="arcs", choices=["arcs", "adamw", "sgd", "muon"])
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--tf32", type=int, default=1, help="allow TF32 matmuls on Ampere+ GPUs (1 = yes)")
    ap.add_argument("--lr", type=float, default=None, help="baselines (Muon: lr of the Muon part)")
    ap.add_argument("--wd", type=float, default=0.01, help="decoupled weight decay (AdamW, Muon)")
    ap.add_argument("--lr_aux", type=float, default=1e-3, help="Muon: Adam lr for head, biases, BN")
    # ARCs-LSR1
    ap.add_argument("--history", type=int, default=50)
    ap.add_argument("--mu_min", type=float, default=1.0)
    ap.add_argument("--eta2", type=float, default=0.5)
    ap.add_argument("--delta_mode", default="code")
    ap.add_argument("--double_grad", action="store_true")
    ap.add_argument("--diag_power", type=float, default=0.0)
    ap.add_argument("--diag_every", type=int, default=1)
    ap.add_argument("--vec_dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--log_steps_every", type=int, default=1, help="ARCs: keep every k-th step record")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cifar10"))
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.opt != "arcs" and a.lr is None:
        a.lr = {"adamw": 1e-3, "sgd": 0.1, "muon": 0.02}[a.opt]
    dev = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                       ("cpu" if a.device == "auto" else a.device))
    if dev.type == "cuda" and a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    tag = run_tag(a)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    Xtr, Ttr, Xte, Tte = load_data(a.data, dev, a.n_fake)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    model = resnet18_cifar().to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    if a.opt == "arcs":
        opt = ARCsLSR1Optim(model.parameters(), y_mode="same", history=a.history, mu_min=a.mu_min, eta2=a.eta2,
                            delta_mode=a.delta_mode, double_grad=a.double_grad, diag_power=a.diag_power,
                            diag_every=a.diag_every, vec_dtype=getattr(torch, a.vec_dtype))
    elif a.opt == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    elif a.opt == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9)
    else:
        opt = ModuleMuon.for_model(model, model.fc, lr=a.lr, wd=a.wd, lr_aux=a.lr_aux)

    log = {"args": vars(a), "tag": tag, "device": str(dev), "n_params": n_params, "epochs": [], "steps": [],
           "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None}
    tl, ta = evaluate(model, Xtr, Ttr, mean, std)
    vl, va = evaluate(model, Xte, Tte, mean, std)
    log["epochs"].append({"epoch": 0, "train_loss": tl, "train_acc": ta, "test_loss": vl, "test_acc": va,
                          "time": 0.0, "steps": 0, "grad_epochs": 0.0})
    print(f"[{tag}] {n_params:,} parameters on {dev}", flush=True)

    gen = torch.Generator().manual_seed(1000 + a.seed)
    gen_aug = torch.Generator().manual_seed(3000 + a.seed)
    N = Xtr.shape[0]
    t0, step, samples = time.time(), 0, 0.0
    stop = False
    model.train()
    for ep in range(1, a.epochs + 1):
        perm = torch.randperm(N, generator=gen).to(dev)
        for bi in range(0, N - a.bs + 1, a.bs):
            idx = perm[bi:bi + a.bs]
            xb = Xtr[idx]
            xb = prep(augment(xb, gen_aug) if a.augment else xb, mean, std)
            yb = Ttr[idx]
            if a.opt == "arcs":
                def closure(backward=True):
                    loss = F.cross_entropy(model(xb), yb)
                    if backward:
                        loss.backward()
                    return loss
                loss = float(opt.step(closure))
                info = opt.last_info
                # forward+backward passes + forward-only passes (a forward ~ 1/3 of a forward+backward)
                samples += (info["n_grad"] + info.get("n_fwd", 0) / 3) * a.bs
                if step % a.log_steps_every == 0:
                    log["steps"].append({"step": step, "loss": info["loss"], "rho": info["rho"], "acc": info["acc"],
                                         "mu": info["mu"], "s_norm": info["s_norm"], "delta": info["delta"][0],
                                         "lam_hat_min": info["lam_hat_min"][0], "k_pairs": info["k_pairs"][0]})
            else:
                opt.zero_grad(set_to_none=True)
                lt = F.cross_entropy(model(xb), yb)
                lt.backward()
                opt.step()
                loss = float(lt.detach())
                samples += a.bs
            step += 1
            if not math.isfinite(loss) or (a.max_steps and step >= a.max_steps):
                stop = True
                break
        tl, ta = evaluate(model, Xtr, Ttr, mean, std)
        vl, va = evaluate(model, Xte, Tte, mean, std)
        rec = {"epoch": ep, "train_loss": tl, "train_acc": ta, "test_loss": vl, "test_acc": va,
               "time": time.time() - t0, "steps": step, "grad_epochs": samples / N}
        if a.opt == "arcs":
            rec["mu"] = opt.core.mu
            rec["accepted"] = sum(s["acc"] for s in log["steps"]) / max(len(log["steps"]), 1)
            rec["opt_gflops"] = opt.flops / 1e9
        if dev.type == "cuda":
            rec["max_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 1e9
        log["epochs"].append(rec)
        print(f"[{tag}] ep {ep} train_loss {tl:.4f} test_acc {va:.4f} grad_epochs {rec['grad_epochs']:.1f} "
              f"t {rec['time']:.0f}s", flush=True)
        if stop or not math.isfinite(tl):
            break
    with open(os.path.join(a.out, tag + ".json"), "w") as f:
        json.dump(log, f)
    return log


if __name__ == "__main__":
    main()
