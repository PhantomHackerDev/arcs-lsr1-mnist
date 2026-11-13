"""
ARCs-LSR1: Adaptive Regularization using Cubics with Limited-memory SR1,
following Ranganath, Singhal & Marcia, arXiv:2502.12298 (Algorithm 1).

Implemented on a single flat parameter vector so that the compact
representation, the partial eigendecomposition and the closed-form cubic
step are explicit and can be instrumented.

Paper pieces reproduced
  * compact L-SR1:  B = delta*I + Psi M Psi^T,  Psi = Y - delta*S,
                    M = (D + L + L^T - delta*S^T S)^{-1}
  * orthonormal basis of range(Psi), eig(R M R^T) = P Lhat P^T, U_par = Q P
  * shape-changing norm ||U^T s||_3 -> separable 1-D cubic problems with the
    closed-form solution  s_i = -2 g_i / (lam_i + sqrt(lam_i^2 + 4 mu |g_i|))
  * perpendicular part handled through ||g_perp|| only (alpha*)
  * SR1 skip rule  |s^T(y - Bs)| > eps ||s|| ||y - Bs||
  * ARC ratio test and mu update (eta1, eta2, gamma1, gamma2)

Implementation note (same math, O(n m) memory traffic per iteration):
  [S | Y] live in one preallocated (n_b, 2m) float64 buffer and its Gram
  matrix G = [S Y]^T [S Y] is updated incrementally when a pair is added.
  Instead of a thin QR of the n x k matrix Psi we use the k x k Gram matrix
  Psi^T Psi = V Sigma V^T, so Q = Psi V Sigma^{-1/2}, R = Sigma^{1/2} V^T and
  U_par = Psi W with W = V Sigma^{-1/2} P (k x r). U_par is never formed;
  only products with the buffer are needed.

Knobs added for the study (defaults reproduce the paper's method)
  delta_mode   'paper'  : delta = min generalized eigenvalue of
                          (D+L+L^T, S^T S), clipped to [delta_lo, delta_hi]
                          (literal reading of Algorithm 1, step 4)
               'safe'   : delta = safe_frac * lambda_hat_min when that is
                          positive, otherwise delta_fallback  (strictly below
                          lambda_hat_min, the condition in Erway et al. 2020)
               'bb'     : delta = y^T y / s^T y of the newest pair (clipped)
               'const'  : delta = delta_fallback
  y_mode       'cross'  : y_k = g_{B_{k+1}}(x_{k+1}) - g_{B_k}(x_k)
                          (gradients from two different mini-batches; this is
                          what a torch.optim.LBFGS-style loop with
                          max_iter=1 produces)
               'same'   : y_k = g_{B_k}(x_k + s_k) - g_{B_k}(x_k)
                          (Direction 3: s and y measured on the same batch)
  blocks       list of (start, end) slices -> one independent L-SR1 model and
               one shape-changing-norm subproblem per block (Direction 1).
               A single block over all parameters is the paper's method.
  grad_beta    EMA weight on the gradient used in the cubic model's linear
               term (Direction 2, one interpretation of "momentum").
  stable_cubic use the cancellation-free form of the 1-D minimizer.
"""
from __future__ import annotations

import math

import torch

DT = torch.float64  # all quasi-Newton linear algebra in double precision


def cubic_1d_min(g: torch.Tensor, lam: torch.Tensor, mu: float, stable: bool = True) -> torch.Tensor:
    """Global minimizer of  g*s + 0.5*lam*s^2 + (mu/3)|s|^3  (elementwise).

    Paper form:  s = -2g / (lam + sqrt(lam^2 + 4 mu |g|)).
    When lam < 0 and |g| is small the denominator is a difference of two
    nearly equal numbers (catastrophic cancellation). The equivalent form
    s = -sign(g) (|lam| + sqrt(lam^2 + 4 mu |g|)) / (2 mu)   for lam < 0
    has no cancellation.
    """
    root = torch.sqrt(lam * lam + 4.0 * mu * g.abs())
    paper = -2.0 * g / (lam + root)
    if not stable:
        return paper
    sgn = torch.where(g >= 0, torch.ones_like(g), -torch.ones_like(g))
    neg = -sgn * (lam.abs() + root) / (2.0 * mu)
    return torch.where(lam < 0, neg, paper)


def cubic_1d_value(g, lam, s, mu):
    return g * s + 0.5 * lam * s * s + (mu / 3.0) * s.abs() ** 3


class BlockState:
    """L-SR1 memory and current eigen-decomposition for one parameter block."""

    def __init__(self, start: int, end: int, m: int, delta: float):
        self.start, self.end, self.m = start, end, m
        nb = end - start
        self.buf = torch.zeros(nb, 2 * m, dtype=DT)  # [S | Y] slots
        self.G = torch.zeros(2 * m, 2 * m, dtype=DT)  # Gram of buf
        self.order: list[int] = []  # slots, oldest -> newest
        self.delta = delta
        # current decomposition: U_par = buf @ Wfull, eigenvalues delta + lam_hat
        self.Wfull: torch.Tensor | None = None  # (2m, r)
        self.lam_hat: torch.Tensor | None = None  # (r,)
        # Hessian-momentum model  B~ = gV*I + V diag(lamV) V^T  (V explicit, orthonormal)
        self.V: torch.Tensor | None = None
        self.lamV: torch.Tensor | None = None
        self.gV: float = delta
        self.n_skip = 0
        self.n_add = 0
        self.flops = 0  # dominant optimizer FLOPs (products with n_b-sized arrays)

    @property
    def k(self):
        return len(self.order)

    def push(self, s: torch.Tensor, y: torch.Tensor):
        m = self.m
        if len(self.order) < m:
            j = len(self.order)
        else:
            j = self.order.pop(0)
        self.order.append(j)
        self.buf[:, j] = s
        self.buf[:, m + j] = y
        v = self.buf.T @ torch.stack([s, y], 1)  # (2m, 2)
        self.flops += 2 * self.buf.shape[0] * self.buf.shape[1] * 2
        for col, c in ((j, 0), (m + j, 1)):
            self.G[:, col] = v[:, c]
            self.G[col, :] = v[:, c]

    def newest(self):
        j = self.order[-1]
        return self.buf[:, j], self.buf[:, self.m + j]

    # U_par products without forming U_par
    def Ut(self, v):
        self.flops += 2 * self.buf.shape[0] * self.buf.shape[1]
        return self.Wfull.T @ (self.buf.T @ v)

    def U(self, z):
        self.flops += 2 * self.buf.shape[0] * self.buf.shape[1]
        return self.buf @ (self.Wfull @ z)


class ARCsLSR1:
    def __init__(
        self,
        n_params: int,
        history: int = 10,
        mu0: float = 1.0,
        eta1: float = 0.1,
        eta2: float = 0.75,
        gamma1: float = 1.0,
        gamma2: float = 3.0,
        mu_min: float = 1e-8,
        mu_max: float = 1e12,
        eps_sr1: float = 1e-8,
        delta_mode: str = "safe",
        delta_lo: float = 1e-4,
        delta_hi: float = 1e4,
        delta_fallback: float = 1.0,
        safe_frac: float = 0.5,
        y_mode: str = "cross",
        blocks: list[tuple[int, int]] | None = None,
        grad_beta: float = 0.0,
        stable_cubic: bool = True,
        rank_tol: float = 1e-13,
        mu_per_layer: bool = False,
        hess_beta: float = 0.0,
        rank_cap: int = 0,
        rho_batch: str = "current",
        mu_floor_kappa: float = 0.0,
        y_batch: str = "current",
        use_pairs: bool = True,
    ):
        """mu_per_layer: one mu_l per block, each adapted by its own ratio
                         rho_l = (f(x) - f(x + s_l)) / pred_l, where x + s_l moves
                         only block l (one extra forward pass per block, no
                         Jacobians). Only blocks with rho_l >= eta1 move.
           rho_batch:    'current' measures the ratios on the batch that produced
                         the step; any other value ('next': an independent batch of
                         the same size, 'large': a larger batch that contains the
                         current one) measures them on the batch behind
                         loss_only_val, so a step that only fits the current batch
                         cannot pass. The gradient in the model is always the
                         current batch's.
           use_pairs:    False stores no curvature pairs, so B = delta_fallback * I
                         (ablation: cubic regularization without quasi-Newton curvature).
           y_batch:      'current' : same-batch pairs use the current batch;
                         'rho'     : same-batch pairs use the ratio batch
                         (grad_val), y = g_R(x + s) - g_R(x); its two loss values
                         are reused for the ratio, so no extra forward passes.
           mu_floor_kappa: if > 0, also track a global mu_g (updated by the
                         combined ratio) and keep every mu_l >= mu_g / kappa.
           hess_beta:    alpha in  B~_{t+1} = alpha B~_t + (1 - alpha) B_new,
                         where B_new is the L-SR1 matrix from the current pairs.
                         B~ is kept as gamma~ I + V Lambda V^T and truncated to
                         rank_cap (default 2*history) by largest |eigenvalue|.
        """
        self.n = n_params
        self.m = history
        self.mu_per_layer = mu_per_layer
        self.hess_beta = hess_beta
        self.rank_cap = rank_cap or 2 * history
        self.eta1, self.eta2 = eta1, eta2
        self.gamma1, self.gamma2 = gamma1, gamma2
        self.mu_min, self.mu_max = mu_min, mu_max
        self.eps_sr1 = eps_sr1
        self.delta_mode = delta_mode
        self.delta_lo, self.delta_hi = delta_lo, delta_hi
        self.delta_fallback = delta_fallback
        self.safe_frac = safe_frac
        self.y_mode = y_mode
        self.grad_beta = grad_beta
        self.stable = stable_cubic
        self.rank_tol = rank_tol
        if blocks is None:
            blocks = [(0, n_params)]
        self.blocks = [BlockState(a, b, history, delta_fallback) for a, b in blocks]
        self.mus = [mu0] * len(self.blocks)  # mus[0] is the shared mu when not per-layer
        self.rho_batch = rho_batch
        self.mu_floor_kappa = mu_floor_kappa
        self.y_batch = y_batch
        self.use_pairs = use_pairs
        self.mu_g = mu0  # global mu used only for the mu_l floor
        self._prev_g = None
        self._prev_s = None
        self._prev_accepted = False
        self._g_ema = None
        self._ema_t = 0
        self._last_s = None
        self._fallback = 0
        self.k = 0

    # ------------------------------------------------------------------ SR1 pieces
    def _choose_delta(self, blk: BlockState, A: torch.Tensor, StS: torch.Tensor) -> tuple[float, float]:
        """Return (delta, lambda_hat_min); lambda_hat_min is the smallest
        generalized eigenvalue of (D+L+L^T) u = lambda S^T S u."""
        k = StS.shape[0]
        lam_hat_min = float("nan")
        try:
            ridge = 1e-12 * float(StS.diagonal().mean())
            Lc = torch.linalg.cholesky(StS + ridge * torch.eye(k, dtype=DT))
            Li = torch.linalg.inv(Lc)
            Cm = Li @ A @ Li.T
            lam_hat_min = float(torch.linalg.eigvalsh(0.5 * (Cm + Cm.T))[0])
        except Exception:
            pass

        mode = self.delta_mode
        if mode == "paper":
            d = lam_hat_min if math.isfinite(lam_hat_min) else self.delta_fallback
        elif mode == "safe":
            d = self.safe_frac * lam_hat_min if (math.isfinite(lam_hat_min) and lam_hat_min > 0) else self.delta_fallback
        elif mode == "bb":
            s, y = blk.newest()
            sy = float(s @ y)
            d = float(y @ y) / sy if sy > 0 else blk.delta
        elif mode == "const":
            d = self.delta_fallback
        else:
            raise ValueError(mode)
        if mode != "const":
            d = min(max(d, self.delta_lo), self.delta_hi)
        return d, lam_hat_min

    def _decompose(self, blk: BlockState) -> dict:
        """delta, U_par, Lambda_hat for one block (Alg. 1 steps 3-8)."""
        info = {"k": blk.k, "lam_hat_min": float("nan"), "cond_minv": float("nan"),
                "cond_psi": float("nan"), "eig_max_abs": 0.0, "bad": 0}
        if blk.k == 0:
            blk.Wfull, blk.lam_hat = None, None
            info["delta"] = blk.delta
            return info
        m = blk.m
        Si = torch.tensor(blk.order)
        Yi = Si + m
        G = blk.G
        StS = G[Si][:, Si]
        SY = G[Si][:, Yi]  # S^T Y, chronological
        YtY = G[Yi][:, Yi]
        L = torch.tril(SY, -1)
        A = torch.diag(torch.diagonal(SY)) + L + L.T
        delta, lam_hat_min = self._choose_delta(blk, A, StS)
        blk.delta = delta
        info["delta"] = delta
        info["lam_hat_min"] = lam_hat_min

        Minv = A - delta * StS
        sv = torch.linalg.svdvals(Minv)
        info["cond_minv"] = float(sv[0] / sv[-1]) if float(sv[-1]) > 0 else float("inf")

        # Psi^T Psi from Gram blocks; orthonormal basis of range(Psi)
        PtP = YtY - delta * (SY + SY.T) + delta * delta * StS
        PtP = 0.5 * (PtP + PtP.T)
        sig, V = torch.linalg.eigh(PtP)
        smax = float(sig[-1])
        keep = sig > self.rank_tol * max(smax, 1e-300)
        info["cond_psi"] = math.sqrt(smax / float(sig[keep][0])) if bool(keep.any()) else float("inf")
        if not bool(keep.any()):
            info["bad"] = 1
            blk.Wfull, blk.lam_hat = None, None
            return info
        sig, V = sig[keep], V[:, keep]
        Rt = (V * sig.sqrt())  # R^T  (k x r), R = Sigma^{1/2} V^T
        try:
            X = torch.linalg.solve(Minv, Rt)  # M R^T
            T = Rt.T @ X  # R M R^T
            T = 0.5 * (T + T.T)
            lam_hat, P = torch.linalg.eigh(T)
            if not (torch.isfinite(lam_hat).all() and torch.isfinite(P).all()):
                raise FloatingPointError
        except Exception:
            info["bad"] = 1
            blk.Wfull, blk.lam_hat = None, None
            return info
        W = (V / sig.sqrt()) @ P  # U_par = Psi W   (k x r)
        Wfull = torch.zeros(2 * m, W.shape[1], dtype=DT)
        Wfull[Yi] = W
        Wfull[Si] = -delta * W
        blk.Wfull, blk.lam_hat = Wfull, lam_hat
        info["eig_max_abs"] = float(lam_hat.abs().max())
        return info

    def _Bv(self, blk: BlockState, v: torch.Tensor) -> torch.Tensor:
        out = blk.delta * v
        if blk.Wfull is not None:
            out = out + blk.U(blk.lam_hat * blk.Ut(v))
        return out

    def _add_pair(self, blk: BlockState, s: torch.Tensor, y: torch.Tensor) -> bool:
        """SR1 skip test with the current B of this block, then push."""
        r = y - self._Bv(blk, s)
        lhs = abs(float(s @ r))
        rhs = self.eps_sr1 * float(s.norm()) * float(r.norm())
        if not (lhs > rhs) or not math.isfinite(lhs):
            blk.n_skip += 1
            return False
        blk.push(s, y)
        blk.n_add += 1
        return True

    # ------------------------------------------------------------------ Hessian momentum
    def _momentum(self, blk: BlockState):
        """B~ <- alpha B~ + (1 - alpha) B_new, compressed to rank_cap.

        Both terms are  gamma I + (orthonormal basis) diag(.) (basis)^T, so the
        sum lives on span[V, U_new]; we orthonormalize that small basis, form the
        projected (r+k)x(r+k) matrix, eigendecompose it and keep the rank_cap
        components with the largest |eigenvalue|.
        """
        a = self.hess_beta
        if blk.Wfull is not None:
            U_new = blk.buf @ blk.Wfull
            blk.flops += 2 * blk.buf.shape[0] * blk.buf.shape[1] * blk.Wfull.shape[1]
            lam_new = blk.lam_hat
        else:
            U_new = torch.zeros(blk.end - blk.start, 0, dtype=DT)
            lam_new = torch.zeros(0, dtype=DT)
        if blk.V is None:
            blk.V, blk.lamV, blk.gV = U_new, lam_new, blk.delta
            return
        r = blk.V.shape[1]
        W = torch.cat([blk.V, U_new], 1)
        if W.shape[1] == 0:
            blk.gV = a * blk.gV + (1 - a) * blk.delta
            return
        G = W.T @ W
        blk.flops += 2 * W.shape[0] * W.shape[1] ** 2
        e, E = torch.linalg.eigh(0.5 * (G + G.T))
        keep = e > 1e-12 * float(e[-1])
        Z = E[:, keep] / e[keep].sqrt()  # Q = W Z orthonormal
        Cv = Z.T @ G[:, :r]  # Q^T V
        Cu = Z.T @ G[:, r:]  # Q^T U_new
        K = a * (Cv * blk.lamV) @ Cv.T + (1 - a) * (Cu * lam_new) @ Cu.T
        kap, P = torch.linalg.eigh(0.5 * (K + K.T))
        idx = torch.argsort(kap.abs(), descending=True)[: self.rank_cap]
        blk.V = W @ (Z @ P[:, idx])
        blk.flops += 2 * W.shape[0] * W.shape[1] * len(idx)
        blk.lamV = kap[idx]
        blk.gV = a * blk.gV + (1 - a) * blk.delta

    # ------------------------------------------------------------------ step
    def _block_step(self, blk: BlockState, g: torch.Tensor, mu: float | None = None):
        """Closed-form minimizer of the block cubic model (shape-changing norm).
        Uses the L-SR1 model, or the Hessian-momentum model when hess_beta > 0.
        Returns (s, model_value, diag)."""
        mu = self.mus[0] if mu is None else mu
        if self.hess_beta > 0 and blk.V is not None:
            has = blk.V.shape[1] > 0
            def Ut(v):
                blk.flops += 2 * blk.V.numel()
                return blk.V.T @ v

            def U(z):
                blk.flops += 2 * blk.V.numel()
                return blk.V @ z
            lam_hat, d = blk.lamV, blk.gV
        else:
            has = blk.Wfull is not None
            Ut, U, lam_hat, d = blk.Ut, blk.U, blk.lam_hat, blk.delta
        gnorm2 = float(g @ g)
        diag = {}
        if has:
            gp = Ut(g)
            lam = d + lam_hat
            sp = cubic_1d_min(gp, lam, mu, self.stable)
            m_par = float(cubic_1d_value(gp, lam, sp, mu).sum())
            gperp2 = max(gnorm2 - float(gp @ gp), 0.0)
            diag["lam_min"] = float(min(float(lam.min()), d))
            diag["lam_max"] = float(max(float(lam.max()), d))
            diag["n_neg"] = int((lam < 0).sum())
        else:
            gp, sp, m_par, gperp2 = None, None, 0.0, gnorm2
            diag["lam_min"] = d
            diag["lam_max"] = d
            diag["n_neg"] = int(d < 0)
        gperp = math.sqrt(gperp2)
        alpha = 2.0 / (d + math.sqrt(d * d + 4.0 * mu * gperp)) if gperp > 0 else 0.0
        m_perp = -alpha * gperp2 + 0.5 * d * alpha * alpha * gperp2 + (mu / 3.0) * (alpha * gperp) ** 3
        s = -alpha * g
        if has:
            # s = U sp - alpha (I - U U^T) g  =  -alpha g + U (alpha gp + sp)
            s = s + U(alpha * gp + sp)
        return s, m_par + m_perp, diag

    def _secant_residual(self, blk: BlockState) -> float:
        """||B s_k - y_k|| / ||y_k|| for the newest stored pair, where B is the
        model used for the step (L-SR1, or B~ with Hessian momentum)."""
        if blk.k == 0:
            return float("nan")
        s, y = blk.newest()
        if self.hess_beta > 0 and blk.V is not None:
            Bs = blk.gV * s + (blk.V @ (blk.lamV * (blk.V.T @ s)) if blk.V.shape[1] else 0.0)
        elif blk.Wfull is not None:
            Bs = blk.delta * s + blk.buf @ (blk.Wfull @ (blk.lam_hat * (blk.Wfull.T @ (blk.buf.T @ s))))
        else:
            Bs = blk.delta * s
        return float((Bs - y).norm() / y.norm())

    @property
    def flops(self) -> int:
        return sum(b.flops for b in self.blocks)

    def _mu_next(self, mu: float, rho: float) -> float:
        if rho > self.eta2:
            mu *= 0.5
        elif rho >= self.eta1:
            mu *= 0.5 * (1.0 + self.gamma1)
        else:
            mu *= 0.5 * (self.gamma1 + self.gamma2)
        return min(max(mu, self.mu_min), self.mu_max)

    @property
    def mu(self) -> float:
        return self.mus[0]

    @mu.setter
    def mu(self, v: float):
        self.mus = [v] * len(self.blocks)

    def _push_pairs(self, s: torch.Tensor, y: torch.Tensor):
        if not self.use_pairs:
            return
        for blk in self.blocks:
            sb = s[blk.start:blk.end]
            if float(sb.abs().max()) == 0.0:  # block did not move: no curvature pair
                continue
            self._add_pair(blk, sb, y[blk.start:blk.end])

    def step(self, x: torch.Tensor, loss_and_grad, loss_only, loss_only_val=None, grad_val=None) -> dict:
        """One ARCs-LSR1 iteration on one mini-batch.

        x                flat float32 parameters (updated in place)
        loss_and_grad(x) -> (float loss, float32 grad) on the CURRENT batch
        loss_only(x)     -> float loss on the CURRENT batch
        loss_only_val(x) -> float loss on the ratio batch (rho_batch != 'current')
        grad_val(x)      -> (float loss, float32 grad) on the ratio batch (y_batch='rho')
        """
        self.k += 1
        f0, g32 = loss_and_grad(x)
        g = g32.to(DT)
        n_grad, n_fwd = 1, 0
        nb = len(self.blocks)

        # cross-batch pair from the previous iteration (paper / LBFGS-style loop)
        if self.y_mode == "cross" and self._prev_g is not None and self._prev_accepted:
            self._push_pairs(self._prev_s, g - self._prev_g)

        # linear term of the model (Direction 2, gradient EMA)
        if self.grad_beta > 0:
            self._ema_t += 1
            if self._g_ema is None:
                self._g_ema = torch.zeros_like(g)
            self._g_ema.mul_(self.grad_beta).add_(g, alpha=1 - self.grad_beta)
            g_model = self._g_ema / (1 - self.grad_beta ** self._ema_t)
        else:
            g_model = g

        per_layer_mu = self.mu_per_layer and nb > 1
        mu_used = list(self.mus) if per_layer_mu else [self.mus[0]] * nb
        infos, steps, preds = [], [], []
        for i, blk in enumerate(self.blocks):
            inf = self._decompose(blk)
            if self.hess_beta > 0:
                self._momentum(blk)
            sb, mb, dg = self._block_step(blk, g_model[blk.start:blk.end], mu_used[i])
            inf.update(dg)
            inf["sec_res"] = self._secant_residual(blk)
            infos.append(inf)
            steps.append(sb)
            preds.append(-mb)
        s_full = torch.cat(steps)
        finite_step = bool(torch.isfinite(s_full).all())

        rho_b = [float("nan")] * nb
        y_same = None  # same-batch curvature pair, measured along s_pairs
        use_val = self.rho_batch != "current" and loss_only_val is not None
        y_val = self.y_mode == "same" and self.y_batch == "rho" and use_val and grad_val is not None
        if per_layer_mu:
            # Aditya's suggestion: per-layer ratio from f with only layer l moved.
            # f_test is the batch the ratios are measured on (current or next).
            f_test = loss_only_val if use_val else loss_only
            f0_t = f0
            if y_val and finite_step:
                f0_t, gv0 = grad_val(x)
                n_grad += 1
            elif use_val:
                f0_t = loss_only_val(x)
                n_fwd += 1
            f_b = [float("inf")] * nb
            for i, blk in enumerate(self.blocks):
                if not finite_step or preds[i] <= 0:
                    rho_b[i] = -float("inf")
                    continue
                xl = x.clone()
                xl[blk.start:blk.end] += steps[i].to(x.dtype)
                f_b[i] = f_test(xl)
                rho_b[i] = (f0_t - f_b[i]) / preds[i]
                n_fwd += 1
            A = [i for i in range(nb) if rho_b[i] >= self.eta1]

            def masked(idx):
                out = torch.zeros_like(s_full)
                for i in idx:
                    b = self.blocks[i]
                    out[b.start:b.end] = steps[i]
                return out

            s_take = masked(A)
            rho_comb = -float("inf")
            if A:
                x_trial = x + s_take.to(x.dtype)
                if y_val:
                    f1, gv1 = grad_val(x_trial)  # y and ratio on the ratio batch
                    y_same = gv1.to(DT) - gv0.to(DT)
                    n_grad += 1
                else:
                    if self.y_mode == "same":
                        f1, g1_32 = loss_and_grad(x_trial)  # y on the current batch
                        y_same = g1_32.to(DT) - g
                        n_grad += 1
                    if use_val:
                        f1 = loss_only_val(x_trial)
                        n_fwd += 1
                    elif self.y_mode != "same":
                        f1 = loss_only(x_trial)
                        n_fwd += 1
                pred = sum(preds[i] for i in A)
                rho_comb = (f0_t - f1) / pred if (math.isfinite(f1) and pred > 0) else -float("inf")
            else:
                pred = sum(preds)
            rho = rho_comb
            accepted = rho_comb >= self.eta1
            s_pairs = s_take  # where y was measured (same-batch mode)
            self._fallback = 0
            if A and not accepted:
                # the block-diagonal model ignores cross-layer curvature, so the
                # combined step can fail although each layer passed on its own:
                # fall back to the single best layer (its f is already known)
                ib = max(A, key=lambda i: f0_t - f_b[i])
                s_take = masked([ib])
                x_trial = x + s_take.to(x.dtype)
                pred, rho, accepted = preds[ib], rho_b[ib], True
                self._fallback = 1
            for i in range(nb):
                # layers that moved together share the outcome of the combined step
                r_i = min(rho_b[i], rho_comb) if i in A else rho_b[i]
                self.mus[i] = self._mu_next(self.mus[i], r_i)
            if self.mu_floor_kappa > 0:
                # global mu from the ratio of the FULL step (all layers, including the
                # ones that failed on their own), as in the shared-mu variant: a bad
                # layer raises mu_g and, through the floor, damps every layer.
                rho_full = -float("inf")
                if finite_step and sum(preds) > 0:
                    f_full = f_test(x + s_full.to(x.dtype))
                    n_fwd += 1
                    rho_full = (f0_t - f_full) / sum(preds)
                self.mu_g = self._mu_next(self.mu_g, rho_full)
                floor = self.mu_g / self.mu_floor_kappa
                self.mus = [max(m_, floor) for m_ in self.mus]
        else:
            s_take = s_full
            pred = sum(preds)
            x_trial = x + s_full.to(x.dtype)
            f0_t = f0
            if not finite_step:
                f1 = float("inf")
            elif y_val:
                # y and both ratio values on the ratio batch
                f0_t, gv0 = grad_val(x)
                f1, gv1 = grad_val(x_trial)
                y_same = gv1.to(DT) - gv0.to(DT)
                n_grad += 2
            else:
                if self.y_mode == "same":
                    f1, g1_32 = loss_and_grad(x_trial)
                    y_same = g1_32.to(DT) - g
                    n_grad += 1
                if use_val:
                    f0_t = loss_only_val(x)
                    f1 = loss_only_val(x_trial)
                    n_fwd += 2
                elif self.y_mode != "same":
                    f1 = loss_only(x_trial)
                    n_fwd += 1
            rho = (f0_t - f1) / pred if (finite_step and math.isfinite(f1) and pred > 0) else -float("inf")
            accepted = rho >= self.eta1
            self.mus = [self._mu_next(self.mus[0], rho)] * nb
            s_pairs = s_full
            self._fallback = 0

        if accepted:
            x.copy_(x_trial)
        # same-batch pair (Direction 3). Curvature info is valid whether or not
        # the step is accepted (Nocedal & Wright, Alg. 6.2).
        if self.y_mode == "same" and y_same is not None:
            self._push_pairs(s_pairs, y_same)

        self._prev_g = g
        self._prev_s = s_take if accepted else None
        self._prev_accepted = accepted
        self._last_s = s_full

        def col(key):
            return [i[key] for i in infos]

        return {
            "loss": f0, "rho": rho if math.isfinite(rho) else -1e9, "acc": int(accepted),
            "mu": max(mu_used), "mu_b": mu_used,
            "rho_b": [r if math.isfinite(r) else -1e9 for r in rho_b],
            "s_norm": float(s_take.norm()) if finite_step else float("nan"),
            "s_norm_b": [float(s.norm()) if finite_step else float("nan") for s in steps],
            "g_norm": float(g.norm()), "pred": pred, "n_grad": n_grad, "n_fwd": n_fwd,
            "fallback": self._fallback, "mu_g": self.mu_g,
            "delta": col("delta"), "lam_min": col("lam_min"), "lam_max": col("lam_max"),
            "n_neg": col("n_neg"), "lam_hat_min": col("lam_hat_min"), "cond_minv": col("cond_minv"),
            "cond_psi": col("cond_psi"), "eig_max_abs": col("eig_max_abs"), "bad": col("bad"),
            "k_pairs": col("k"), "sec_res": col("sec_res"),
        }

    # ------------------------------------------------------------------ diagnostics
    def curvature_probe(self, hvp, s: torch.Tensor) -> dict:
        """Compare what B believes with a large-batch Hessian H.

        q_B(s) = s^T B s/||s||^2   vs   q_H(s) = s^T H s/||s||^2
        and, for the most negative eigen-direction u of B (if any),
        lam_min(B)                 vs   u^T H u.
        hvp(v_float64) -> H v (float64) on a fixed large probe batch.
        """
        res = {}
        ss = float(s @ s)
        if ss == 0 or not math.isfinite(ss):
            return res
        Bs = torch.cat([self._Bv(b, s[b.start:b.end]) for b in self.blocks])
        Hs = hvp(s)
        res["qB_s"] = float(s @ Bs) / ss
        res["qH_s"] = float(s @ Hs) / ss
        best = None
        for b in self.blocks:
            if b.Wfull is None:
                continue
            lam = b.delta + b.lam_hat
            i = int(torch.argmin(lam))
            if float(lam[i]) < 0 and (best is None or float(lam[i]) < best[0]):
                best = (float(lam[i]), b, i)
        if best is not None:
            lam_val, b, i = best
            u = torch.zeros(self.n, dtype=DT)
            u[b.start:b.end] = b.buf @ b.Wfull[:, i]
            u = u / u.norm()
            res["lamB_neg"] = lam_val
            res["qH_neg"] = float(u @ hvp(u))
        return res
