from __future__ import annotations
import math
import torch
from torch.optim import Optimizer


def manual_adam_sequence(weight: float, gradients, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    m = 0.0
    v = 0.0
    w = float(weight)
    rows = []
    for t, g in enumerate(gradients, start=1):
        g = float(g)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        signed_step = -lr * m_hat / (math.sqrt(v_hat) + eps)
        w_new = w + signed_step
        rows.append({
            "step": t, "gradient": g, "m": m, "v": v,
            "m_hat": m_hat, "v_hat": v_hat,
            "signed_step": signed_step, "weight_after": w_new,
        })
        w = w_new
    return rows


class AdamNoBiasCorrection(Optimizer):
    """Adam with exactly one change: no first/second-moment bias correction."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, (b1, b2), eps = group["lr"], group["betas"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(b1).add_(g, alpha=1-b1)
                v.mul_(b2).addcmul_(g, g, value=1-b2)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-lr)
        return loss
