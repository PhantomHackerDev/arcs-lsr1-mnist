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

# 10. delta rule 'safe_keep': keeps the previous delta when lambda_hat_min <= 0 -----
#     (a pair with negative curvature makes lambda_hat_min < 0; 'safe' resets to the fallback)
o_s = ARCsLSR1(4, history=3, delta_mode="safe")
o_k = ARCsLSR1(4, history=3, delta_mode="safe_keep")
A_pos, St = torch.eye(2, dtype=DT), torch.eye(2, dtype=DT)
A_neg = torch.diag(torch.tensor([1.0, -1.0], dtype=DT))
res = []
for o in (o_s, o_k):
    blk = o.blocks[0]
    d1, lam1 = o._choose_delta(blk, A_pos, St)
    blk.delta = d1
    d2, lam2 = o._choose_delta(blk, A_neg, St)
    res.append((d1, lam1, d2, lam2))
check("'safe' and 'safe_keep' agree when lambda_hat_min > 0",
      res[0][0] == res[1][0] and abs(res[0][0] - o_s.safe_frac) < 1e-9, f"(delta {res[0][0]:.12g})")
check("lambda_hat_min < 0: 'safe' resets to delta_fallback, 'safe_keep' keeps the previous delta",
      res[0][2] == o_s.delta_fallback and res[1][2] == res[1][0] and res[1][3] < 0,
      f"(safe {res[0][2]:g}, safe_keep {res[1][2]:g}, lambda_hat_min {res[1][3]:g})")

# 11. delta rule 'safe_lm': LM damping of the pairs when lambda_hat_min <= 0 ------------
torch.manual_seed(11)
n11, m11 = 8, 3
H11 = torch.diag(torch.linspace(-2.0, 3.0, n11, dtype=DT))  # indefinite curvature
S11 = torch.randn(n11, m11, dtype=DT)
Y11 = H11 @ S11


def dense_sr1_from(B, S_, Y_):
    for j in range(S_.shape[1]):
        r = Y_[:, j] - B @ S_[:, j]
        B = B + torch.outer(r, r) / (r @ S_[:, j])
    return B


def dense_B(o):
    blk = o.blocks[0]
    return torch.stack([o._Bv(blk, e) for e in torch.eye(n11, dtype=DT)], 1)


def build(mode, S_, Y_, delta_prev=0.3):
    o = ARCsLSR1(n11, history=m11, delta_mode=mode)
    blk = o.blocks[0]
    for j in range(S_.shape[1]):
        blk.push(S_[:, j], Y_[:, j])
    blk.delta = delta_prev
    return o, o._decompose(blk)


o_lm, inf_lm = build("safe_lm", S11, Y11)
lh11 = inf_lm["lam_hat_min"]
gam, lm11 = inf_lm["delta"], inf_lm["lm"]
dps = gam - lm11
SY11 = S11.T @ Y11
A11 = torch.diag(torch.diagonal(SY11)) + torch.tril(SY11, -1) + torch.tril(SY11, -1).T
Psi11 = Y11 - dps * S11
B_exp = gam * torch.eye(n11, dtype=DT) + Psi11 @ torch.linalg.solve(A11 - dps * S11.T @ S11, Psi11.T)
B_lm = dense_B(o_lm)
check("'safe_lm': lambda_hat_min < 0 gives lam = delta_prev/safe_frac - lambda_hat_min and keeps delta",
      lh11 < 0 and abs(lm11 - (0.3 / 0.5 - lh11)) < 1e-12 and gam == 0.3, f"(lambda_hat_min {lh11:.3f}, lam {lm11:.3f})")
check("'safe_lm': B = delta I + Psi~ (A - (delta - lam) S^T S)^{-1} Psi~^T, Psi~ = Y - (delta - lam) S",
      torch.allclose(B_lm, B_exp, atol=1e-10), f"(max err {float((B_lm - B_exp).abs().max()):.1e})")
B_rec = dense_sr1_from(dps * torch.eye(n11, dtype=DT), S11, Y11) + lm11 * torch.eye(n11, dtype=DT)
check("'safe_lm': B = SR1(B0 = (delta - lam) I) + lam I  (SR1 shift invariance)",
      torch.allclose(B_lm, B_rec, atol=1e-9), f"(max err {float((B_lm - B_rec).abs().max()):.1e})")
ev = torch.linalg.eigvalsh(0.5 * (B_lm + B_lm.T))
check("'safe_lm': B >= delta I (positive definite)", float(ev[0]) >= gam - 1e-9, f"(lambda_min(B) {float(ev[0]):.4f})")
Y11p = torch.diag(torch.linspace(0.5, 3.0, n11, dtype=DT)) @ S11  # positive curvature
o_a, i_a = build("safe", S11, Y11p)
o_b, i_b = build("safe_lm", S11, Y11p)
check("'safe_lm' == 'safe' when lambda_hat_min > 0", i_b["lm"] == 0.0 and torch.equal(dense_B(o_a), dense_B(o_b)))


# 12. double_grad: the model gradient averages two batches at the same point -------------
def quad(D, c):
    return (lambda xv: (float(0.5 * (D * (xv.to(DT) - c) ** 2).sum()), (D * (xv.to(DT) - c)).float()),
            lambda xv: float(0.5 * (D * (xv.to(DT) - c) ** 2).sum()))


torch.manual_seed(12)
n12 = 10
D12 = torch.linspace(0.5, 4.0, n12, dtype=DT)
batches = [quad(D12, torch.randn(n12, dtype=DT)) for _ in range(10)]
x_ref, x_dg = torch.randn(n12), None
x_dg = x_ref.clone()
o_ref = ARCsLSR1(n12, history=3, y_mode="same")
o_dg = ARCsLSR1(n12, history=3, y_mode="same", double_grad_ratio=True)
same = True
for k in range(12):  # identical batches: the averages equal the single batch, bitwise
    lg_, lo_ = batches[0]
    d1, d2 = o_ref.step(x_ref, lg_, lo_), o_dg.step(x_dg, lg_, lo_)
    same &= bool(torch.equal(x_ref, x_dg)) and d1["rho"] == d2["rho"]
check("double_grad(_ratio) with identical batches reproduces the default bitwise", same)
o_dg = ARCsLSR1(n12, history=3, y_mode="same", double_grad_ratio=True, mu0=0.01)  # small mu0: some rejections
xk = torch.randn(n12)
grad_ok, rho_ok, accs = True, True, []
for k in range(1, 10):
    lg_, lo_ = batches[k]
    lg_p, lo_p = batches[k - 1]
    x0 = xk.clone()
    d = o_dg.step(xk, lg_, lo_)
    accs.append(d["acc"])
    if k > 1:
        g_exp = 0.5 * (lg_(x0)[1].to(DT) + lg_p(x0)[1].to(DT))
        grad_ok &= torch.allclose(o_dg._last_g_model, g_exp, atol=1e-6)
        s_ = o_dg._last_s
        fbar = lambda v: 0.5 * (lo_(v) + lo_p(v))  # noqa: E731
        r_exp = (fbar(x0) - fbar(x0 + s_.float())) / d["pred"]
        rho_ok &= abs(d["rho"] - r_exp) < 1e-6 * max(1.0, abs(r_exp))
check("double_grad: model gradient = (g_B_k(x_k) + g_B_{k-1}(x_k)) / 2, after accepted and rejected steps",
      grad_ok and 0 in accs[:-1] and 1 in accs[:-1], f"(accepted: {accs})")
check("double_grad_ratio: rho is measured on the two-batch average loss", rho_ok)

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
