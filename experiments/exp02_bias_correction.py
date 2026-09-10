from __future__ import annotations
import math
import pandas as pd
import matplotlib.pyplot as plt
import torch
from src.optimizers import AdamNoBiasCorrection
from src.utils import ARTIFACTS

GRADS = [0.10, -0.20, 0.05, -0.10, 0.15]


def correction_ratio(step: int, beta1=.9, beta2=.999):
    """Raw Adam adaptive term / bias-corrected adaptive term, ignoring epsilon."""
    return math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)


def first_below(threshold=.01, consecutive=20, max_steps=20000):
    ok = 0
    for t in range(1, max_steps + 1):
        rel = abs(correction_ratio(t) - 1.0)
        ok = ok + 1 if rel < threshold else 0
        if ok >= consecutive:
            return t - consecutive + 1
    raise RuntimeError("threshold not reached")


def _first_sustained(values, threshold=.01, consecutive=20):
    ok = 0
    for step, value in values:
        ok = ok + 1 if value < threshold else 0
        if ok >= consecutive:
            return int(step) - consecutive + 1
    return None


def run(max_steps=5000):
    p_c = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    p_n = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    opt_c = torch.optim.Adam([p_c], lr=1e-3, betas=(.9,.999), eps=1e-8)
    opt_n = AdamNoBiasCorrection([p_n], lr=1e-3, betas=(.9,.999), eps=1e-8)

    rows=[]
    prev_c, prev_n = 1.0, 1.0
    for t in range(1, max_steps + 1):
        g = GRADS[(t-1) % len(GRADS)]
        for p,opt in [(p_c,opt_c),(p_n,opt_n)]:
            opt.zero_grad(set_to_none=True)
            p.grad = torch.tensor([g],dtype=torch.float64)
            opt.step()
        uc = p_c.item()-prev_c; un=p_n.item()-prev_n
        prev_c,prev_n=p_c.item(),p_n.item()
        measured_gap = abs(uc-un) / (abs(uc) + 1e-30)
        theory_gap = abs(correction_ratio(t)-1.0)
        rows.append({
            "step":t,"gradient":g,"weight_corrected":p_c.item(),"weight_uncorrected":p_n.item(),
            "signed_update_corrected":uc,"signed_update_uncorrected":un,
            "abs_update_gap":abs(uc-un),"relative_update_gap":measured_gap,
            "correction_multiplier":correction_ratio(t),"relative_multiplier_gap":theory_gap,
        })
    long_df=pd.DataFrame(rows)
    first20=long_df.head(20).copy()
    first20.to_csv(ARTIFACTS/"02_bias_correction_first20.csv",index=False)
    long_df.to_csv(ARTIFACTS/"02_bias_correction_long_horizon.csv",index=False)

    fig,ax=plt.subplots(figsize=(8,4.8))
    ax.plot(first20.step,first20.weight_corrected,marker='o',label='Adam: bias corrected')
    ax.plot(first20.step,first20.weight_uncorrected,marker='o',label='Adam: no bias correction')
    ax.set(xlabel='Step',ylabel='Weight value',title='First 20 steps: bias correction changes Adam trajectory')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/"02_bias_correction_trajectory.png",dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.8))
    ax.plot(first20.step,first20.relative_update_gap*100,marker='o',label='measured update gap')
    ax.plot(first20.step,first20.relative_multiplier_gap*100,marker='.',label='theoretical multiplier gap')
    ax.axhline(1.0,linestyle='--',label='1% materiality threshold')
    ax.set(xlabel='Step',ylabel='Relative gap (%)',title='First 20 steps: corrected vs uncorrected Adam')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/"02_bias_correction_gap.png",dpi=180); plt.close(fig)

    theoretical_step=first_below(.01,20)
    measured_step=_first_sustained(long_df[['step','relative_update_gap']].itertuples(index=False,name=None),.01,20)

    fig,ax=plt.subplots(figsize=(9,5.0))
    ax.plot(long_df.step,long_df.relative_update_gap*100,label='measured relative update gap')
    ax.plot(long_df.step,long_df.relative_multiplier_gap*100,label='theoretical multiplier gap',alpha=.8)
    ax.axhline(1.0,linestyle='--',label='1% threshold')
    ax.axvline(theoretical_step,linestyle=':',label=f'theory: {theoretical_step}')
    ax.axvline(measured_step,linestyle='-.',label=f'measured: {measured_step}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set(xlabel='Step (log scale)',ylabel='Relative gap, % (log scale)',title='Bias-correction materiality over 5,000 steps')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/"02_bias_correction_long_horizon.png",dpi=180); plt.close(fig)

    return {
        "materiality_threshold":0.01,
        "consecutive_steps":20,
        "theoretical_first_sustained_below_threshold":theoretical_step,
        "measured_first_sustained_below_threshold":measured_step,
        "interpretation":"The analytical multiplier becomes <1% at step 3916; the actual update stream follows within a few steps."
    }

if __name__=='__main__': print(run())
