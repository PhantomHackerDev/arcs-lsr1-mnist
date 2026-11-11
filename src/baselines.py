"""Modern first-/second-order baselines on a single flat parameter vector.

All three take the flat leaf tensor `xp` and the list of parameter shapes
(weight, bias, weight, bias, ...), so they can treat each weight as a matrix
(conv kernels are flattened to (out_channels, in_channels*k*k)).
AdamW is torch.optim.AdamW and needs no wrapper.

Muon     Jordan et al. 2024 (kellerjordan.github.io/posts/muon). Hidden weight
         matrices: Nesterov momentum, then orthogonalised with the quintic
         Newton-Schulz iteration (coefficients 3.4445, -4.7750, 2.0315, 5 steps),
         scaled by sqrt(max(1, rows/cols)). As recommended, the output layer and
         all biases use AdamW instead (lr_aux).
Shampoo  Gupta, Koren, Singer 2018, in the common modern form (Anil et al. 2020):
         EMA Kronecker statistics L = E[G G^T], R = E[G^T G]; direction
         L^{-1/4} M R^{-1/4} for the momentum M, with the per-layer step size
         grafted from Adam (norm of the Adam update). Roots are recomputed
         every `precond_every` steps by eigendecomposition with relative
         damping; Adam is used until the first preconditioner exists and for
         biases.
SophiaG  Liu et al. 2023 (github.com/Liuhong99/Sophia). Momentum m, diagonal
         Hessian EMA h from the Gauss-Newton-Bartlett estimator (gradient of the
         loss at labels sampled from the model, squared, every k steps), update
         -lr * sign(m) * min(|m| / (rho * batch * h), 1).
All use decoupled weight decay `wd`.
"""
from __future__ import annotations

import math

import torch


def segments(shapes):
    out, i = [], 0
    for sh in shapes:
        n = math.prod(sh)
        out.append((i, i + n, tuple(sh)))
        i += n
    return out


def as_matrix(v, shape):
    return v.view(shape[0], -1)


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7, dtype=torch.float32) -> torch.Tensor:
    """Approximate U V^T of G = U S V^T (Muon's quintic iteration).

    Same operation sequence as torch.optim.Muon (fused addmm), so with dtype=torch.bfloat16 it
    reproduces torch exactly. torch runs it in bfloat16 (a GPU speed choice); float32 is the
    default here (more precise, and faster on CPU)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(dtype=dtype, copy=True)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X.div_(X.norm().clamp(min=eps))
    for _ in range(steps):
        A = X @ X.T
        B = torch.addmm(A, A, A, beta=b, alpha=c)  # b*A + c*A@A
        X = torch.addmm(X, B, X, beta=a)  # a*X + B@X
    if transposed:
        X = X.T
    return X.float()


class _AdamState:
    """Elementwise Adam moments for one segment (used directly or for grafting)."""

    def __init__(self, n, betas=(0.9, 0.999), eps=1e-8):
        self.m = torch.zeros(n)
        self.v = torch.zeros(n)
        self.b1, self.b2 = betas
        self.eps = eps

    def update(self, g, t):
        self.m.mul_(self.b1).add_(g, alpha=1 - self.b1)
        self.v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
        mh = self.m / (1 - self.b1 ** t)
        vh = self.v / (1 - self.b2 ** t)
        return mh / (vh.sqrt() + self.eps), mh


class FlatMuon:
    def __init__(self, xp, shapes, lr=0.02, momentum=0.95, wd=0.01, lr_aux=1e-3, ns_steps=5,
                 ns_dtype=torch.float32):
        self.xp, self.lr, self.mom, self.wd, self.lr_aux, self.ns = xp, lr, momentum, wd, lr_aux, ns_steps
        self.ns_dtype = ns_dtype
        segs = segments(shapes)
        last_w = max(k for k, (_, _, sh) in enumerate(segs) if len(sh) >= 2)
        self.muon = [(s, e, sh) for k, (s, e, sh) in enumerate(segs) if len(sh) >= 2 and k != last_w]
        self.aux = [(s, e, sh) for k, (s, e, sh) in enumerate(segs) if not (len(sh) >= 2 and k != last_w)]
        self.buf = {s: torch.zeros(e - s) for s, e, _ in self.muon}
        self.adam = {s: _AdamState(e - s) for s, e, _ in self.aux}
        self.t = 0

    def zero_grad(self):
        self.xp.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        x, g = self.xp.data, self.xp.grad
        for s, e, sh in self.muon:
            buf = self.buf[s]
            # Nesterov momentum in torch.optim.Muon's "lerp" form; it equals (1 - mom) times the
            # classic g + mom * (mom * buf + g), and the scale is removed by the normalisation
            buf.lerp_(g[s:e], 1 - self.mom)
            upd = g[s:e].lerp(buf, self.mom)
            M = as_matrix(upd, sh)
            O = newton_schulz5(M, self.ns, dtype=self.ns_dtype)
            scale = max(1.0, M.shape[0] / M.shape[1]) ** 0.5
            x[s:e].mul_(1 - self.lr * self.wd).add_(O.reshape(-1), alpha=-self.lr * scale)
        for s, e, sh in self.aux:
            u, _ = self.adam[s].update(g[s:e], self.t)
            x[s:e].mul_(1 - self.lr_aux * self.wd).add_(u, alpha=-self.lr_aux)


class FlatShampoo:
    def __init__(self, xp, shapes, lr=1e-3, betas=(0.9, 0.999), wd=0.01, precond_every=10,
                 damping=1e-6, max_dim=2048):
        self.xp, self.lr, self.wd, self.T, self.damp = xp, lr, wd, precond_every, damping
        self.b1, self.b2 = betas
        self.segs = segments(shapes)
        self.adam = {s: _AdamState(e - s, betas) for s, e, _ in self.segs}
        self.mats = {}
        for s, e, sh in self.segs:
            if len(sh) >= 2:
                r, c = sh[0], math.prod(sh[1:])
                if max(r, c) <= max_dim:
                    self.mats[s] = {"L": torch.zeros(r, r), "R": torch.zeros(c, c), "Li": None, "Ri": None}
        self.t = 0

    def zero_grad(self):
        self.xp.grad = None

    @staticmethod
    def _inv_root(A, damping, p=4):
        A = 0.5 * (A + A.T)
        lam, Q = torch.linalg.eigh(A.double())
        lam = lam.clamp(min=0) + damping * max(float(lam.max()), 1e-30)
        return ((Q * lam.pow(-1.0 / p)) @ Q.T).float()

    @torch.no_grad()
    def step(self):
        self.t += 1
        x, g = self.xp.data, self.xp.grad
        for s, e, sh in self.segs:
            adam_dir, mh = self.adam[s].update(g[s:e], self.t)
            direction = adam_dir
            st = self.mats.get(s)
            if st is not None:
                G = as_matrix(g[s:e], sh)
                st["L"].mul_(self.b2).add_(G @ G.T, alpha=1 - self.b2)
                st["R"].mul_(self.b2).add_(G.T @ G, alpha=1 - self.b2)
                if self.t % self.T == 0:
                    st["Li"] = self._inv_root(st["L"], self.damp)
                    st["Ri"] = self._inv_root(st["R"], self.damp)
                if st["Li"] is not None:
                    D = (st["Li"] @ as_matrix(mh, sh) @ st["Ri"]).reshape(-1)
                    dn = float(D.norm())
                    if dn > 0:  # graft: Shampoo direction, Adam step length (per layer)
                        direction = D * (float(adam_dir.norm()) / dn)
            x[s:e].mul_(1 - self.lr * self.wd).add_(direction, alpha=-self.lr)


class FlatSophiaG:
    def __init__(self, xp, lr=3e-4, betas=(0.965, 0.99), rho=0.04, wd=0.01, k=10):
        self.xp, self.lr, self.rho, self.wd, self.k = xp, lr, rho, wd, k
        self.b1, self.b2 = betas
        n = xp.numel()
        self.m = torch.zeros(n)
        self.h = torch.zeros(n)
        self.t = 0

    def zero_grad(self):
        self.xp.grad = None

    def wants_hessian(self) -> bool:
        """True on the steps where the caller should run update_hessian (official schedule)."""
        return self.t % self.k == self.k - 1

    @torch.no_grad()
    def update_hessian(self, g_sampled: torch.Tensor):
        """g_sampled: gradient of the loss at labels sampled from the model (GNB estimator)."""
        self.h.mul_(self.b2).addcmul_(g_sampled, g_sampled, value=1 - self.b2)

    @torch.no_grad()
    def step(self, batch_size: int):
        self.t += 1
        x, g = self.xp.data, self.xp.grad
        self.m.mul_(self.b1).add_(g, alpha=1 - self.b1)
        x.mul_(1 - self.lr * self.wd)
        ratio = (self.m.abs() / (self.rho * batch_size * self.h + 1e-15)).clamp(max=1.0)
        x.addcmul_(self.m.sign(), ratio, value=-self.lr)
