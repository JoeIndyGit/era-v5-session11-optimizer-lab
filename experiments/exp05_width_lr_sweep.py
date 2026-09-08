from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from src.config import CFG
from src.data import make_teacher_dataset, make_batch_indices
from src.model import StudentMLP
from src.schedules import cosine_factor
from src.utils import ARTIFACTS, seed_everything, device

WIDTHS=[256,512,1024]
COARSE=[0.0005,0.001,0.002,0.004,0.008,0.016,0.032]
FINE=[0.004,0.005,0.006,0.008,0.010,0.012]
SEEDS=[41,42,43]


def train_one(width:int,peak_lr:float,steps=200,seed=42):
    dev=device(); seed_everything(seed)
    xtr,ytr,xv,yv=make_teacher_dataset(device=dev)
    batches=make_batch_indices(steps)
    model=StudentMLP(width).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=peak_lr,weight_decay=CFG.weight_decay)
    for st,idx_cpu in enumerate(batches):
        lr=peak_lr*cosine_factor(st,steps,CFG.warmup_steps,CFG.min_lr_factor)
        for pg in opt.param_groups: pg['lr']=lr
        idx=idx_cpu.to(dev)
        opt.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(model(xtr[idx]),ytr[idx])
        loss.backward(); opt.step()
    with torch.no_grad():
        val=nn.functional.cross_entropy(model(xv),yv).item()
    return float(loss.item()),float(val)


def quadratic_log_minimum(lrs,losses):
    x=np.log(np.asarray(lrs,dtype=float)); y=np.asarray(losses,dtype=float)
    a,b,c=np.polyfit(x,y,2)
    if a <= 0:
        i=int(np.argmin(y)); return float(lrs[i]), (float(a),float(b),float(c))
    xstar=-b/(2*a)
    xstar=float(np.clip(xstar,x.min(),x.max()))
    return float(np.exp(xstar)), (float(a),float(b),float(c))


def run():
    coarse_rows=[]
    for width in WIDTHS:
        for lr in COARSE:
            tr,va=train_one(width,lr,seed=42)
            coarse_rows.append({'phase':'coarse','width':width,'peak_lr':lr,'seed':42,'train_loss':tr,'val_loss':va})
    coarse=pd.DataFrame(coarse_rows)
    coarse.to_csv(ARTIFACTS/'05_width_lr_coarse.csv',index=False)

    fine_rows=[]
    for width in WIDTHS:
        for lr in FINE:
            for seed in SEEDS:
                tr,va=train_one(width,lr,seed=seed)
                fine_rows.append({'phase':'fine','width':width,'peak_lr':lr,'seed':seed,'train_loss':tr,'val_loss':va})
    fine=pd.DataFrame(fine_rows)
    fine.to_csv(ARTIFACTS/'05_width_lr_fine_all_seeds.csv',index=False)
    summary=fine.groupby(['width','peak_lr']).agg(val_loss_mean=('val_loss','mean'),val_loss_std=('val_loss','std')).reset_index()
    summary.to_csv(ARTIFACTS/'05_width_lr_fine_summary.csv',index=False)

    minima=[]
    for width in WIDTHS:
        g=summary[summary.width==width].sort_values('peak_lr')
        lrstar,coeff=quadratic_log_minimum(g.peak_lr.values,g.val_loss_mean.values)
        discrete=float(g.loc[g.val_loss_mean.idxmin(),'peak_lr'])
        minima.append({'width':width,'discrete_min_lr':discrete,'quadratic_min_lr':lrstar,'fit_a':coeff[0],'fit_b':coeff[1],'fit_c':coeff[2]})
    mins=pd.DataFrame(minima)

    alpha,log_c=np.polyfit(np.log(mins.width.values.astype(float)),np.log(mins.quadratic_min_lr.values),1)
    pred4096=float(np.exp(log_c+alpha*np.log(4096.0)))
    mins['power_law_alpha']=alpha
    mins['predicted_lr_width4096']=pred4096
    mins.to_csv(ARTIFACTS/'05_width_lr_minima.csv',index=False)

    fig,ax=plt.subplots(figsize=(9,5.2))
    for width,g in summary.groupby('width'):
        ax.errorbar(g.peak_lr,g.val_loss_mean,yerr=g.val_loss_std,marker='o',capsize=3,label=f'width {width}')
        m=mins[mins.width==width].iloc[0]
        ax.axvline(m.quadratic_min_lr,linestyle=':',alpha=.45)
        ax.scatter([m.quadratic_min_lr],[np.interp(np.log(m.quadratic_min_lr),np.log(g.peak_lr),g.val_loss_mean)],s=65)
    ax.set_xscale('log')
    ax.set(xlabel='Peak learning rate (log scale)',ylabel='Validation loss',title='Width × learning-rate sweep (mean ± SD over 3 seeds)')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/'05_width_lr_sweep.png',dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,4.8))
    ax.scatter(mins.width,mins.quadratic_min_lr,s=70,label='fitted minima')
    x=np.array([256,512,1024,2048,4096],dtype=float)
    y=np.exp(log_c+alpha*np.log(x))
    ax.plot(x,y,label=f'power law: lr ∝ width^{alpha:.3f}')
    ax.scatter([4096],[pred4096],marker='x',s=100,label=f'4096 prediction ≈ {pred4096:.4g}')
    ax.set_xscale('log',base=2); ax.set_yscale('log')
    ax.set(xlabel='Width',ylabel='Best peak learning rate',title='Extrapolating the LR optimum to width 4096')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/'05_width_lr_extrapolation.png',dpi=180); plt.close(fig)

    return {
        'minima':minima,
        'power_law_alpha':float(alpha),
        'predicted_lr_width4096':pred4096,
        'confidence':'moderate',
        'confidence_reason':'Three widths and three seeds show a smooth trend, but width 4096 is a 4× extrapolation beyond the largest measured width and the optimum depends on this parameterization/task.'
    }

if __name__=='__main__': print(run())
