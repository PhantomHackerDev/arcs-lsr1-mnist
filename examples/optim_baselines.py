"""Baselines for nn.Module training loops, matching the flat-vector versions used on MNIST / F-MNIST.

ModuleMuon: the same update as src/baselines.FlatMuon (Nesterov momentum in torch.optim.Muon's lerp form,
5 Newton-Schulz iterations, scale max(1, rows/cols)^0.5, decoupled weight decay). Muon is applied to every weight
with >= 2 dimensions (conv filters reshaped to out x (in*kh*kw)) except the classifier head; the head, biases and
normalization parameters use Adam (lr_aux). torch.optim.Muon only accepts 2-D parameters, hence this wrapper.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from baselines import newton_schulz5  # noqa: E402


class ModuleMuon(torch.optim.Optimizer):
    def __init__(self, muon_params, aux_params, lr=0.02, momentum=0.95, wd=0.01, lr_aux=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, ns_steps=5):
        groups = [{"params": list(muon_params), "muon": True, "lr": lr},
                  {"params": list(aux_params), "muon": False, "lr": lr_aux}]
        super().__init__(groups, dict(momentum=momentum, wd=wd, betas=betas, eps=eps, ns_steps=ns_steps))
        self.t = 0

    @classmethod
    def for_model(cls, model, head, **kw):
        """Muon for all >= 2-D weights except those of `head` (the classifier); Adam for the rest."""
        head_ids = {id(p) for p in head.parameters()}
        mu, aux = [], []
        for p in model.parameters():
            (mu if (p.ndim >= 2 and id(p) not in head_ids) else aux).append(p)
        return cls(mu, aux, **kw)

    @torch.no_grad()
    def step(self, closure=None):
        self.t += 1
        for grp in self.param_groups:
            lr, wd = grp["lr"], grp["wd"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                g, st = p.grad, self.state[p]
                if grp["muon"]:
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(p)
                    buf = st["buf"]
                    buf.lerp_(g, 1 - grp["momentum"])
                    upd = g.lerp(buf, grp["momentum"])
                    M = upd.reshape(p.shape[0], -1)
                    O = newton_schulz5(M, grp["ns_steps"])
                    scale = max(1.0, M.shape[0] / M.shape[1]) ** 0.5
                    p.mul_(1 - lr * wd).add_(O.reshape(p.shape), alpha=-lr * scale)
                else:  # Adam with decoupled weight decay (as _AdamState in src/baselines.py)
                    b1, b2 = grp["betas"]
                    if "m" not in st:
                        st["m"], st["v"] = torch.zeros_like(p), torch.zeros_like(p)
                    st["m"].mul_(b1).add_(g, alpha=1 - b1)
                    st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
                    mh = st["m"] / (1 - b1 ** self.t)
                    vh = st["v"] / (1 - b2 ** self.t)
                    p.mul_(1 - lr * wd).add_(mh / (vh.sqrt() + grp["eps"]), alpha=-lr)
        return None
