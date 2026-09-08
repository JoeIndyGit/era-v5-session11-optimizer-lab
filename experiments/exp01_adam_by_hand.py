from __future__ import annotations
import pandas as pd
import torch
from src.optimizers import manual_adam_sequence
from src.utils import ARTIFACTS

GRADIENTS = [0.10, -0.20, 0.05, -0.10, 0.15]


def run():
    manual = manual_adam_sequence(1.0, GRADIENTS)
    p = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([p], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    rows = []
    prev = p.detach().item()
    for i, g in enumerate(GRADIENTS, start=1):
        opt.zero_grad(set_to_none=True)
        p.grad = torch.tensor([g], dtype=torch.float64)
        opt.step()
        state = opt.state[p]
        m = state["exp_avg"].item()
        v = state["exp_avg_sq"].item()
        m_hat = m / (1 - 0.9 ** i)
        v_hat = v / (1 - 0.999 ** i)
        signed_step = p.detach().item() - prev
        prev = p.detach().item()
        r = manual[i-1]
        rows.append({
            "step": i, "gradient": g,
            "manual_m": r["m"], "torch_m": m, "abs_err_m": abs(r["m"]-m),
            "manual_v": r["v"], "torch_v": v, "abs_err_v": abs(r["v"]-v),
            "manual_m_hat": r["m_hat"], "torch_m_hat": m_hat, "abs_err_m_hat": abs(r["m_hat"]-m_hat),
            "manual_v_hat": r["v_hat"], "torch_v_hat": v_hat, "abs_err_v_hat": abs(r["v_hat"]-v_hat),
            "manual_signed_step": r["signed_step"], "torch_signed_step": signed_step,
            "abs_err_step": abs(r["signed_step"]-signed_step),
            "manual_weight_after": r["weight_after"], "torch_weight_after": p.detach().item(),
        })
    df = pd.DataFrame(rows)
    out = ARTIFACTS / "01_adam_verification.csv"
    df.to_csv(out, index=False)
    max_err = float(df[[c for c in df if c.startswith("abs_err")]].to_numpy().max())
    assert max_err < 1e-12, f"Manual Adam mismatch: {max_err}"
    return {"max_absolute_error": max_err, "rows": rows}

if __name__ == "__main__":
    print(run())
