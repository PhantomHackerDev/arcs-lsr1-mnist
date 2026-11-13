"""Sanity checks for the ARCs-LSR1 building blocks (run: python src/test_math.py)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arcs_lsr1 import DT, ARCsLSR1, cubic_1d_min, cubic_1d_value  # noqa: E402

torch.manual_seed(0)
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


# 1. closed-form 1-D cubic minimizer vs brute force -------------------------
g = torch.randn(2000, dtype=DT) * torch.logspace(-6, 1, 2000, dtype=DT)
lam = torch.randn(2000, dtype=DT) * 3
mu = 0.7
s = cubic_1d_min(g, lam, mu)
grid = torch.linspace(-30, 30, 600001, dtype=DT)
worst = 0.0
for i in range(0, 2000, 50):
    v = cubic_1d_value(g[i], lam[i], grid, mu).min()
    worst = max(worst, float(cubic_1d_value(g[i], lam[i], s[i], mu) - v))
check("1-D cubic closed form is the global minimizer", worst < 1e-6, f"(max excess {worst:.2e})")

# 2. float32 cancellation in the paper's formula for lam<0, small |g| --------
g32 = torch.tensor([1e-6, 1e-5, 1e-4], dtype=torch.float32)
lam32 = torch.tensor([-50.0, -50.0, -50.0], dtype=torch.float32)
ref = cubic_1d_min(g32.double(), lam32.double(), 1.0, stable=True)
naive = cubic_1d_min(g32, lam32, 1.0, stable=False).double()
stab = cubic_1d_min(g32, lam32, 1.0, stable=True).double()
rel_naive = ((naive - ref).abs() / ref.abs()).max().item()
rel_stab = ((stab - ref).abs() / ref.abs()).max().item()
print(f"       float32, lam=-50, |g| in [1e-6,1e-4]: paper form rel.err {rel_naive:.2e}, "
      f"stable form rel.err {rel_stab:.2e}")
print(f"       paper form gives {naive.tolist()}  (true {ref.tolist()})")
check("stable form fixes float32 cancellation", rel_stab < 1e-5 and rel_naive > 1e-2)

# 3. compact representation == recursive SR1 from B0 = delta I ---------------
n, m = 40, 6
A = torch.randn(n, n, dtype=DT)
H = 0.5 * (A + A.T)  # indefinite "Hessian"
opt = ARCsLSR1(n, history=m, delta_mode="const", delta_fallback=0.8)
blk = opt.blocks[0]
pairs = []
for _ in range(m + 3):  # overfill: also tests the circular buffer
    s_ = torch.randn(n, dtype=DT)
    y_ = H @ s_ + 0.01 * torch.randn(n, dtype=DT)
    blk.push(s_, y_)
    pairs.append((s_, y_))
pairs = pairs[-m:]
opt._decompose(blk)
Ud = blk.buf @ blk.Wfull
orth = float((Ud.T @ Ud - torch.eye(Ud.shape[1], dtype=DT)).abs().max())
check("U_par has orthonormal columns", orth < 1e-9, f"(max dev {orth:.1e})")
Bc = blk.delta * torch.eye(n, dtype=DT) + Ud @ torch.diag(blk.lam_hat) @ Ud.T
B = blk.delta * torch.eye(n, dtype=DT)
for s_, y_ in pairs:
    r = y_ - B @ s_
    B = B + torch.outer(r, r) / (s_ @ r)
err = float((Bc - B).abs().max() / B.abs().max())
check("compact L-SR1 == recursive SR1 (last m pairs)", err < 1e-8, f"(rel err {err:.1e})")
sec = float((Bc @ pairs[-1][0] - pairs[-1][1]).norm() / pairs[-1][1].norm())
check("newest secant equation B s_k = y_k holds", sec < 1e-8, f"(rel resid {sec:.1e})")
opt_q = ARCsLSR1(n, history=m, delta_mode="const", delta_fallback=0.8)
bq = opt_q.blocks[0]
qp = []
for _ in range(m):
    s_ = torch.randn(n, dtype=DT)
    bq.push(s_, H @ s_)
    qp.append((s_, H @ s_))
opt_q._decompose(bq)
Uq = bq.buf @ bq.Wfull
Bq = bq.delta * torch.eye(n, dtype=DT) + Uq @ torch.diag(bq.lam_hat) @ Uq.T
secq = max(float((Bq @ s_ - y_).norm() / y_.norm()) for s_, y_ in qp)
check("all secant equations hold on quadratic data", secq < 1e-8, f"(max rel resid {secq:.1e})")
check("SR1 matrix is indefinite here (negative curvature captured)", float(torch.linalg.eigvalsh(Bc)[0]) < 0)
Bv_err = float((opt._Bv(blk, pairs[0][0]) - Bc @ pairs[0][0]).norm() / (Bc @ pairs[0][0]).norm())
check("matrix-free B v matches dense B v", Bv_err < 1e-10, f"(rel err {Bv_err:.1e})")

# 4. step == global minimizer of the shape-changing-norm cubic model ---------
gvec = torch.randn(n, dtype=DT)
opt.mu = 0.3
s_star, mval, _ = opt._block_step(blk, gvec)
gp = gvec - Ud @ (Ud.T @ gvec)
r_ = Ud.shape[1]
Qfull, _ = torch.linalg.qr(torch.cat([Ud, gp[:, None], torch.randn(n, n, dtype=DT)], 1))
U = Qfull[:, :n].clone()
U[:, r_] *= torch.sign(U[:, r_] @ gp)


def model(sv):
    return float(gvec @ sv + 0.5 * sv @ Bc @ sv + opt.mu / 3 * (U.T @ sv).abs().pow(3).sum())


check("model value bookkeeping matches", abs(model(s_star) - mval) < 1e-8 * abs(mval), f"({model(s_star):.6f} vs {mval:.6f})")
better = 0
for _ in range(3000):
    d = torch.randn(n, dtype=DT) * 10 ** torch.empty(1).uniform_(-4, 0).item()
    if model(s_star + d) < model(s_star) - 1e-12:
        better += 1
check("no random perturbation improves the model", better == 0, f"({better}/3000 improved)")

# 5. delta = lambda_hat_min makes M^{-1} singular ----------------------------
S = torch.stack([p[0] for p in pairs], 1)
Y = torch.stack([p[1] for p in pairs], 1)
SY = S.T @ Y
A_sym = torch.diag(torch.diagonal(SY)) + torch.tril(SY, -1) + torch.tril(SY, -1).T
StS = S.T @ S
Li = torch.linalg.inv(torch.linalg.cholesky(StS))
lh = torch.linalg.eigvalsh(Li @ A_sym @ Li.T)
for frac, label in [(1.0, "delta = lam_hat_min (Alg. 1 literal)"), (0.5, "delta = 0.5*lam_hat_min")]:
    c = torch.linalg.cond(A_sym - frac * float(lh[0]) * StS).item()
    print(f"       {label:38s}: lam_hat_min={float(lh[0]):+.3f}  cond(M^-1) = {c:.2e}")
check("cond(M^-1) explodes at delta = lam_hat_min",
      torch.linalg.cond(A_sym - float(lh[0]) * StS).item() > 1e12)

# 6. Hessian momentum: B~ == alpha*B~_old + (1-alpha)*B_new without truncation ---
n, m = 30, 3
optm = ARCsLSR1(n, history=m, delta_mode="const", delta_fallback=0.7, hess_beta=0.6, rank_cap=100)
bm = optm.blocks[0]
A0 = torch.randn(n, n, dtype=DT)
Hm = 0.5 * (A0 + A0.T)


def dense_sr1(b):
    Uu = b.buf @ b.Wfull
    return b.delta * torch.eye(n, dtype=DT) + Uu @ torch.diag(b.lam_hat) @ Uu.T


Bt = None
worst = 0.0
for it in range(6):
    s_ = torch.randn(n, dtype=DT)
    bm.push(s_, Hm @ s_ + 0.05 * torch.randn(n, dtype=DT))
    optm._decompose(bm)
    Bn = dense_sr1(bm)
    optm._momentum(bm)
    Bt = Bn if Bt is None else 0.6 * Bt + 0.4 * Bn
    Bm_ = bm.gV * torch.eye(n, dtype=DT) + bm.V @ torch.diag(bm.lamV) @ bm.V.T
    worst = max(worst, float((Bm_ - Bt).abs().max() / Bt.abs().max()))
check("Hessian momentum == exact matrix EMA (no truncation)", worst < 1e-9, f"(max rel err {worst:.1e})")
orthV = float((bm.V.T @ bm.V - torch.eye(bm.V.shape[1], dtype=DT)).abs().max())
check("momentum basis V is orthonormal", orthV < 1e-9, f"(max dev {orthV:.1e})")

# 7. per-layer mu on a separable toy problem: f decreases, mus adapt separately --
torch.manual_seed(1)
nA, nB = 20, 20
DA = torch.linspace(0.5, 2.0, nA, dtype=DT)
DB = torch.linspace(5.0, 50.0, nB, dtype=DT)  # stiffer block
Dall = torch.cat([DA, DB])
xs = torch.randn(nA + nB)


def lg(xv):
    xd = xv.to(DT)
    return float(0.5 * (Dall * xd * xd).sum()), (Dall * xd).float()


def lo(xv):
    xd = xv.to(DT)
    return float(0.5 * (Dall * xd * xd).sum())


optl = ARCsLSR1(nA + nB, history=4, y_mode="same", blocks=[(0, nA), (nA, nA + nB)], mu_per_layer=True)
f_start = lo(xs)
for _ in range(40):
    optl.step(xs, lg, lo)
f_end = lo(xs)
check("per-layer mu: objective decreases on toy quadratic", f_end < 1e-3 * f_start,
      f"({f_start:.3e} -> {f_end:.3e}, mus={[f'{v:.2g}' for v in optl.mus]})")

# 8. next-batch ratio test and mu_l floor run and keep their invariants ----------
xs2 = torch.randn(nA + nB)
optn = ARCsLSR1(nA + nB, history=4, y_mode="same", blocks=[(0, nA), (nA, nA + nB)], mu_per_layer=True,
                rho_batch="next", mu_floor_kappa=10.0)
f_start = lo(xs2)
floor_ok = True
for _ in range(40):
    d = optn.step(xs2, lg, lo, lo)  # deterministic toy: the "next batch" is the same function
    floor_ok &= all(m_ >= optn.mu_g / 10.0 - 1e-12 for m_ in optn.mus)
check("next-batch ratio + mu floor: objective decreases", lo(xs2) < 1e-3 * f_start,
      f"({f_start:.3e} -> {lo(xs2):.3e})")
check("mu_l >= mu_g / kappa holds every iteration", floor_ok)

# 9. large ratio batch / curvature pairs on it ----------------------------------
# (a) if the "large" batch is the same function as the current batch, the trajectory
#     must be bitwise identical to the default, on the full and per-layer paths
for kw in ({}, {"blocks": [(0, nA), (nA, nA + nB)], "mu_per_layer": True, "mu_floor_kappa": 10.0}):
    for yb in ("current", "rho"):
        x_ref = torch.randn(nA + nB)
        x_new = x_ref.clone()
        o_ref = ARCsLSR1(nA + nB, history=4, y_mode="same", **kw)
        o_new = ARCsLSR1(nA + nB, history=4, y_mode="same", rho_batch="large", y_batch=yb, **kw)
        same = True
        for _ in range(30):
            d1 = o_ref.step(x_ref, lg, lo)
            d2 = o_new.step(x_new, lg, lo, lo, lg)
            same &= bool(torch.equal(x_ref, x_new)) and d1["rho"] == d2["rho"] and d1["mu_b"] == d2["mu_b"]
        check(f"ratio batch == current batch reproduces the default ({'per-layer' if kw else 'full'}, "
              f"y_batch={yb})", same)


# (b) with a different ratio function: the ratio uses it, and y_batch='rho' builds y from it
def make_q(D):
    return (lambda xv: (float(0.5 * (D * xv.to(DT) ** 2).sum()), (D * xv.to(DT)).float()),
            lambda xv: float(0.5 * (D * xv.to(DT) ** 2).sum()))


Dr = Dall * 1.5
lg_r, lo_r = make_q(Dr)
xq = torch.randn(nA + nB)
oq = ARCsLSR1(nA + nB, history=4, y_mode="same", rho_batch="large", y_batch="rho")
x_before = xq.clone()
d = oq.step(xq, lg, lo, lo_r, lg_r)
S_, Y_ = oq.blocks[0].newest()  # the trial step s and its curvature pair
rho_expect = (lo_r(x_before) - lo_r(x_before + S_.float())) / d["pred"]
check("large ratio batch: rho is measured on the ratio function", abs(d["rho"] - rho_expect) < 1e-9 * abs(rho_expect),
      f"(rho {d['rho']:.6f} vs {rho_expect:.6f}; on the current batch it would be "
      f"{(lo(x_before) - lo(x_before + S_.float())) / d['pred']:.6f})")
check("y_batch='rho': stored y = g_R(x+s) - g_R(x)", torch.allclose(Y_, Dr * S_, rtol=1e-5, atol=1e-6),
      f"(max err {float((Y_ - Dr * S_).abs().max()):.1e})")

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
