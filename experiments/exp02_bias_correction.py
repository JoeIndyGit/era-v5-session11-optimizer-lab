from __future__ import annotations
import math
import pandas as pd
import matplotlib.pyplot as plt
import torch
from src.optimizers import AdamNoBiasCorrection
from src.utils import ARTIFACTS

GRADS = [0.10, -0.20, 0.05, -0.10, 0.15]


def correction_ratio(step: int, beta1=.9, beta2=.999):
    return math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)


def first_below(threshold=.01, consecutive=20, max_steps=20000):
    ok = 0
    for t in range(1, max_steps + 1):
        rel = abs(correction_ratio(t) - 1.0)
        ok = ok + 1 if rel < threshold else 0
        if ok >= consecutive:
            return t - consecutive + 1
    raise RuntimeError("threshold not reached")


def run():
    p_c = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    p_n = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    opt_c = torch.optim.Adam([p_c], lr=1e-3, betas=(.9,.999), eps=1e-8)
    opt_n = AdamNoBiasCorrection([p_n], lr=1e-3, betas=(.9,.999), eps=1e-8)
    rows=[]
    prev_c, prev_n = 1.0, 1.0
    for t in range(1, 21):
        g = GRADS[(t-1) % len(GRADS)]
        for p,opt in [(p_c,opt_c),(p_n,opt_n)]:
            opt.zero_grad(set_to_none=True)
            p.grad = torch.tensor([g],dtype=torch.float64)
            opt.step()
        uc = p_c.item()-prev_c; un=p_n.item()-prev_n
        prev_c,prev_n=p_c.item(),p_n.item()
        rows.append({"step":t,"gradient":g,"weight_corrected":p_c.item(),"weight_uncorrected":p_n.item(),
                     "signed_update_corrected":uc,"signed_update_uncorrected":un,
                     "abs_update_gap":abs(uc-un),"correction_multiplier":correction_ratio(t),
                     "relative_multiplier_gap":abs(correction_ratio(t)-1.0)})
    df=pd.DataFrame(rows)
    df.to_csv(ARTIFACTS/"02_bias_correction_first20.csv",index=False)

    fig,ax=plt.subplots(figsize=(8,4.8))
    ax.plot(df.step,df.weight_corrected,marker='o',label='Adam: bias corrected')
    ax.plot(df.step,df.weight_uncorrected,marker='o',label='Adam: no bias correction')
    ax.set(xlabel='Step',ylabel='Weight value',title='Bias correction changes the early Adam trajectory')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/"02_bias_correction_trajectory.png",dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.8))
    ax.plot(df.step,df.relative_multiplier_gap*100,marker='o')
    ax.axhline(1.0,linestyle='--',label='1% materiality threshold')
    ax.set(xlabel='Step',ylabel='Bias-correction multiplier gap (%)',title='First 20 steps: correction remains materially large')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/"02_bias_correction_gap.png",dpi=180); plt.close(fig)

    threshold_step=first_below(.01,20)
    return {"materiality_threshold":0.01,"consecutive_steps":20,"first_sustained_below_threshold":threshold_step}

if __name__=='__main__': print(run())
