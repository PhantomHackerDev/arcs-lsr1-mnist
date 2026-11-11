"""torch.optim interface to ARCs-LSR1.

    import torch.nn as nn, torch.nn.functional as F
    from arcs_optim import ARCsLSR1Optim

    model = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))
    layers = [m for m in model if isinstance(m, nn.Linear)]
    # one parameter group per layer -> one L-SR1 model / cubic subproblem per layer
    opt = ARCsLSR1Optim([{"params": l.parameters()} for l in layers],
                        per_layer=True, y_mode="same", mu_per_layer=True, mu_floor_kappa=1.0)

    for xb, yb in loader:
        def closure(backward=True):
            loss = F.cross_entropy(model(xb), yb)
            if backward:
                loss.backward()
            return loss
        loss = opt.step(closure)

Closure contract (as for torch.optim.LBFGS, plus one keyword):
  * every call within one step() must evaluate the SAME mini-batch: the
    ARC ratio test compares f(x) and f(x + s) on it;
  * it may accept `backward`: with backward=False it only returns the loss.
    A closure without that keyword also works, but every loss-only
    evaluation then pays for a backward pass;
  * gradients are zeroed by the optimizer before each evaluation.
For rho_batch="next", pass closure_val (same contract, another batch).

Parameter groups define the blocks when per_layer=True. All keyword
arguments other than per_layer go to ARCsLSR1 (see arcs_lsr1.py).
Limitation: state_dict() does not include the L-SR1 memory.
"""
from __future__ import annotations

import inspect

import torch

from arcs_lsr1 import ARCsLSR1


class ARCsLSR1Optim(torch.optim.Optimizer):
    def __init__(self, params, per_layer: bool = False, **kwargs):
        super().__init__(params, {})
        self._params = [p for g in self.param_groups for p in g["params"]]
        if any(p.dtype != torch.float32 for p in self._params):
            raise TypeError("ARCsLSR1Optim expects float32 parameters")
        blocks, i = [], 0
        for g in self.param_groups:
            n = sum(p.numel() for p in g["params"])
            blocks.append((i, i + n))
            i += n
        self.core = ARCsLSR1(i, blocks=blocks if per_layer else None, **kwargs)
        self.last_info: dict = {}

    # flat-vector views of the parameters
    def _get(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self._params])

    @torch.no_grad()
    def _set(self, x: torch.Tensor):
        i = 0
        for p in self._params:
            n = p.numel()
            p.copy_(x[i:i + n].view_as(p))
            i += n

    def _grad(self) -> torch.Tensor:
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          for p in self._params])

    @staticmethod
    def _takes_backward(fn) -> bool:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        return "backward" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    def _wrap(self, closure):
        has_kw = self._takes_backward(closure)

        def loss_and_grad(xv):
            self._set(xv)
            for p in self._params:
                p.grad = None
            with torch.enable_grad():
                loss = closure(backward=True) if has_kw else closure()
            return float(loss.detach()), self._grad()

        def loss_only(xv):
            self._set(xv)
            if has_kw:
                with torch.no_grad():
                    loss = closure(backward=False)
            else:  # closure insists on calling backward: let it, then discard the grads
                with torch.enable_grad():
                    loss = closure()
                for p in self._params:
                    p.grad = None
            return float(loss.detach())

        return loss_and_grad, loss_only

    def step(self, closure, closure_val=None):  # noqa: D102  (torch.optim API)
        if closure is None:
            raise ValueError("ARCsLSR1Optim.step requires a closure")
        x = self._get()
        lg, lo = self._wrap(closure)
        lo_val = self._wrap(closure_val)[1] if closure_val is not None else None
        self.last_info = self.core.step(x, lg, lo, lo_val)
        self._set(x)  # x holds the accepted iterate (unchanged if the step was rejected)
        return torch.tensor(self.last_info["loss"])

    @property
    def flops(self) -> int:
        """Dominant optimizer FLOPs so far (products with parameter-sized arrays)."""
        return self.core.flops
