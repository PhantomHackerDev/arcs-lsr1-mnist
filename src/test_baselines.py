"""Checks for the modern baselines (run: python src/test_baselines.py)."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import FlatMuon, FlatShampoo, FlatSophiaG, newton_schulz5, segments  # noqa: E402

torch.manual_seed(0)
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


# 1. Newton-Schulz output is close to the orthogonal polar factor
G = torch.randn(64, 40)
sv = torch.linalg.svdvals(newton_schulz5(G))
check("Newton-Schulz: singular values in Muon's (0.5, 1.3) band", 0.5 < sv.min() and sv.max() < 1.3,
      f"(min {sv.min():.2f}, max {sv.max():.2f})")

# 2a. Newton-Schulz in bfloat16 == torch's own implementation, bitwise
from torch.optim._muon import _zeropower_via_newtonschulz  # noqa: E402

for shp in ((64, 40), (40, 64)):
    Gq = torch.randn(*shp)
    d = float((newton_schulz5(Gq, dtype=torch.bfloat16) - _zeropower_via_newtonschulz(
        Gq, (3.4445, -4.7750, 2.0315), 5, 1e-7).float()).abs().max())
    check(f"Newton-Schulz (bfloat16) == torch implementation, shape {shp}", d == 0.0, f"(max abs diff {d:.1e})")

# 2b. FlatMuon's hidden-matrix update == torch.optim.Muon
shapes = [(16, 12), (16,), (5, 16), (5,)]
segs = segments(shapes)
xp = torch.randn(sum(e - s for s, e, _ in segs), requires_grad=True)
W_ref = xp.detach()[:16 * 12].view(16, 12).clone().requires_grad_(True)
fm = FlatMuon(xp, shapes, lr=0.02, momentum=0.95, wd=0.01, ns_dtype=torch.bfloat16)  # torch's precision
tm = torch.optim.Muon([W_ref], lr=0.02, momentum=0.95, weight_decay=0.01, nesterov=True)
worst = 0.0
for _ in range(5):
    g = torch.randn_like(xp)
    xp.grad = g.clone()
    fm.step()
    W_ref.grad = g[:16 * 12].view(16, 12).clone()
    tm.step()
    worst = max(worst, float((xp.detach()[:16 * 12].view(16, 12) - W_ref.detach()).abs().max()))
check("FlatMuon hidden-layer update == torch.optim.Muon (5 steps, both in bfloat16)", worst < 1e-5, f"(max abs diff {worst:.1e})")

# 3. Shampoo inverse fourth root
A = torch.randn(30, 30)
A = A @ A.T + 0.1 * torch.eye(30)
Ri = FlatShampoo._inv_root(A, 1e-6)
lam_max = float(torch.linalg.eigvalsh(A).max())
err = float((torch.linalg.matrix_power(Ri.double(), 4) @ (A.double() + 1e-6 * lam_max * torch.eye(30, dtype=torch.float64))
             - torch.eye(30, dtype=torch.float64)).abs().max())
check("Shampoo: (L + damping)^(-1/4) is correct", err < 1e-3, f"(max |R^4 L - I| {err:.1e})")

# 4. Sophia: per-coordinate step bounded by lr (plus decoupled weight decay)
xs = torch.randn(200, requires_grad=True)
so = FlatSophiaG(xs, lr=1e-2, wd=0.0)
x_before = xs.detach().clone()
xs.grad = 100 * torch.randn(200)
so.step(batch_size=32)
step_max = float((xs.detach() - x_before).abs().max())
check("Sophia: |update| <= lr per coordinate (clipping)", step_max <= 1e-2 + 1e-7, f"(max {step_max:.2e})")

# 5. every baseline trains a small MLP on a fixed synthetic task
X = torch.randn(512, 20)
yt = (X @ torch.randn(20, 4)).argmax(1)
shapes = [(32, 20), (32,), (4, 32), (4,)]


def loss_of(v, Xb=X, yb=yt):
    W1, b1, W2, b2 = [v[s:e].view(sh) for s, e, sh in segments(shapes)]
    return F.cross_entropy(F.linear(F.relu(F.linear(Xb, W1, b1)), W2, b2), yb)


def init():
    torch.manual_seed(1)
    return torch.cat([torch.randn(32 * 20) * 0.2, torch.zeros(32), torch.randn(4 * 32) * 0.2,
                      torch.zeros(4)]).requires_grad_(True)


for name in ("AdamW", "Muon", "Shampoo", "Sophia"):
    v = init()
    if name == "AdamW":
        opt = torch.optim.AdamW([v], lr=3e-3)
    elif name == "Muon":
        opt = FlatMuon(v, shapes, lr=0.02, lr_aux=3e-3)
    elif name == "Shampoo":
        opt = FlatShampoo(v, shapes, lr=3e-3)
    else:
        opt = FlatSophiaG(v, lr=3e-3)
    l0 = float(loss_of(v))
    for t in range(150):
        opt.zero_grad()
        L = loss_of(v)
        L.backward()
        if name == "Sophia":
            if opt.wants_hessian():
                g_keep = v.grad.clone()
                logits = F.linear(F.relu(F.linear(X, *[v[s:e].view(sh) for s, e, sh in segments(shapes)][:2])),
                                  *[v[s:e].view(sh) for s, e, sh in segments(shapes)][2:])
                ys = torch.distributions.Categorical(logits=logits.detach()).sample()
                (gs,) = torch.autograd.grad(F.cross_entropy(logits, ys), v)
                opt.update_hessian(gs)
                v.grad = g_keep
            opt.step(batch_size=512)
        else:
            opt.step()
    l1 = float(loss_of(v))
    check(f"{name}: training loss decreases on a small MLP", l1 < 0.3 * l0, f"({l0:.3f} -> {l1:.3f})")

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
