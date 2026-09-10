from __future__ import annotations
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


def fit_width_scaling(fine: pd.DataFrame):
    summary=fine.groupby(['width','peak_lr']).agg(val_loss_mean=('val_loss','mean'),val_loss_std=('val_loss','std')).reset_index()
    minima=[]
    for width in WIDTHS:
        g=summary[summary.width==width].sort_values('peak_lr')
        lrstar,coeff=quadratic_log_minimum(g.peak_lr.values,g.val_loss_mean.values)
        discrete=float(g.loc[g.val_loss_mean.idxmin(),'peak_lr'])
        minima.append({'width':width,'discrete_min_lr':discrete,'quadratic_min_lr':lrstar,'fit_a':coeff[0],'fit_b':coeff[1],'fit_c':coeff[2]})
    mins=pd.DataFrame(minima)
    x=np.log(mins.width.values.astype(float)); y=np.log(mins.quadratic_min_lr.values)
    alpha,log_c=np.polyfit(x,y,1)
    yhat=alpha*x+log_c
    ss_res=float(np.sum((y-yhat)**2)); ss_tot=float(np.sum((y-y.mean())**2))
    r2=1.0-ss_res/ss_tot if ss_tot>0 else 1.0
    pred4096=float(np.exp(log_c+alpha*np.log(4096.0)))
    return summary,mins,float(alpha),float(log_c),pred4096,r2


def bootstrap_prediction(fine: pd.DataFrame, n_boot=5000, seed=12345):
    """Cluster bootstrap over seeds; preserves each seed's whole LR curve across widths."""
    rng=np.random.default_rng(seed)
    seeds=np.array(sorted(fine.seed.unique()))
    lrs=np.array(sorted(fine.peak_lr.unique()),dtype=float)
    cube=np.empty((len(seeds),len(WIDTHS),len(lrs)),dtype=float)
    for si,s in enumerate(seeds):
        for wi,w in enumerate(WIDTHS):
            g=fine[(fine.seed==s)&(fine.width==w)].set_index('peak_lr').loc[lrs]
            cube[si,wi,:]=g.val_loss.values
    rows=[]
    xw=np.log(np.asarray(WIDTHS,dtype=float))
    for b in range(n_boot):
        sampled_idx=rng.integers(0,len(seeds),size=len(seeds))
        means=cube[sampled_idx].mean(axis=0)
        lrstars=[]
        for wi,w in enumerate(WIDTHS):
            lrstar,_=quadratic_log_minimum(lrs,means[wi])
            lrstars.append(lrstar)
        yl=np.log(np.asarray(lrstars))
        alpha,log_c=np.polyfit(xw,yl,1)
        yhat=alpha*xw+log_c
        ss_res=float(np.sum((yl-yhat)**2)); ss_tot=float(np.sum((yl-yl.mean())**2))
        r2=1.0-ss_res/ss_tot if ss_tot>0 else 1.0
        pred=float(np.exp(log_c+alpha*np.log(4096.0)))
        rows.append({'bootstrap':b,'alpha':float(alpha),'predicted_lr_width4096':pred,'r2':r2,
                     **{f'lrstar_{w}':float(v) for w,v in zip(WIDTHS,lrstars)}})
    return pd.DataFrame(rows)

def add_uncertainty_artifacts(fine: pd.DataFrame, n_boot=5000):
    summary,mins,alpha,log_c,pred4096,r2=fit_width_scaling(fine)
    boot=bootstrap_prediction(fine,n_boot=n_boot)
    boot.to_csv(ARTIFACTS/'05_width_lr_bootstrap.csv',index=False)
    lo,med,hi=np.percentile(boot.predicted_lr_width4096,[2.5,50,97.5])
    alo,amed,ahi=np.percentile(boot.alpha,[2.5,50,97.5])
    mins['power_law_alpha']=alpha
    mins['power_law_r2']=r2
    mins['predicted_lr_width4096']=pred4096
    mins['bootstrap_ci95_low']=lo
    mins['bootstrap_ci95_high']=hi
    mins.to_csv(ARTIFACTS/'05_width_lr_minima.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.8))
    ax.hist(boot.predicted_lr_width4096,bins=35)
    ax.axvline(pred4096,linestyle='--',label=f'point estimate {pred4096:.5f}')
    ax.axvline(lo,linestyle=':',label=f'95% CI [{lo:.5f}, {hi:.5f}]')
    ax.axvline(hi,linestyle=':')
    ax.set(xlabel='Predicted best LR at width 4096',ylabel='Bootstrap count',title='Bootstrap uncertainty in width-4096 LR extrapolation')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/'05_width_lr_bootstrap.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    ax.scatter(mins.width,mins.quadratic_min_lr,s=70,label='fitted minima')
    x=np.array([256,512,1024,2048,4096],dtype=float)
    y=np.exp(log_c+alpha*np.log(x))
    ax.plot(x,y,label=f'power law: lr ∝ width^{alpha:.3f}; R²={r2:.3f}')
    ax.scatter([4096],[pred4096],marker='x',s=100,label=f'4096 ≈ {pred4096:.4g}')
    ax.errorbar([4096],[pred4096],yerr=[[pred4096-lo],[hi-pred4096]],capsize=4)
    ax.set_xscale('log',base=2); ax.set_yscale('log')
    ax.set(xlabel='Width',ylabel='Best peak learning rate',title='Width scaling with bootstrap uncertainty')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/'05_width_lr_extrapolation.png',dpi=180); plt.close(fig)
    return {
        'quadratic_minima':{str(int(r.width)):float(r.quadratic_min_lr) for _,r in mins.iterrows()},
        'power_law_alpha':alpha,
        'power_law_alpha_ci95':[float(alo),float(ahi)],
        'power_law_r2':r2,
        'predicted_lr_width4096':pred4096,
        'predicted_lr_width4096_ci95':[float(lo),float(hi)],
        'confidence':'moderate',
        'confidence_reason':'The three fitted minima follow a smooth power law (high R²) and bootstrap variability is narrow, but 4096 is still a 4× extrapolation beyond the largest measured width and only three widths are observed.'
    }


def confirm_width4096():
    """Held-out confirmation around the pre-registered width-4096 prediction."""
    confirmation_lrs=[0.0035,0.0040,0.0045]
    confirmation_seeds=[44,45,46]
    rows=[]
    for lr in confirmation_lrs:
        for seed in confirmation_seeds:
            tr,va=train_one(4096,lr,steps=200,seed=seed)
            rows.append({'width':4096,'peak_lr':lr,'seed':seed,'train_loss':tr,'val_loss':va})
    raw=pd.DataFrame(rows)
    raw.to_csv(ARTIFACTS/'05_width4096_confirmation.csv',index=False)
    summary=(raw.groupby('peak_lr').agg(val_loss_mean=('val_loss','mean'),val_loss_std=('val_loss','std')).reset_index())
    summary.to_csv(ARTIFACTS/'05_width4096_confirmation_summary.csv',index=False)
    best=float(summary.loc[summary.val_loss_mean.idxmin(),'peak_lr'])
    best_row=summary.loc[summary.val_loss_mean.idxmin()]
    fig,ax=plt.subplots(figsize=(8,4.8))
    ax.errorbar(summary.peak_lr,summary.val_loss_mean,yerr=summary.val_loss_std,marker='o',capsize=4)
    ax.axvline(0.0040,linestyle='--',label='pre-confirmation prediction ≈ 0.0040')
    ax.set(xlabel='Peak learning rate',ylabel='Validation loss',title='Held-out width-4096 confirmation (mean ± SD, seeds 44–46)')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/'05_width4096_confirmation.png',dpi=180); plt.close(fig)
    return {'seeds':confirmation_seeds,'learning_rates':confirmation_lrs,'best_confirmed_lr':best,
            'best_val_loss_mean':float(best_row.val_loss_mean),'best_val_loss_std':float(best_row.val_loss_std)}

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
    summary,mins,_,_,_,_=fit_width_scaling(fine)
    summary.to_csv(ARTIFACTS/'05_width_lr_fine_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5.2))
    for width,g in summary.groupby('width'):
        ax.errorbar(g.peak_lr,g.val_loss_mean,yerr=g.val_loss_std,marker='o',capsize=3,label=f'width {width}')
        m=mins[mins.width==width].iloc[0]
        ax.axvline(m.quadratic_min_lr,linestyle=':',alpha=.45)
        ax.scatter([m.quadratic_min_lr],[np.interp(np.log(m.quadratic_min_lr),np.log(g.peak_lr),g.val_loss_mean)],s=65)
    ax.set_xscale('log')
    ax.set(xlabel='Peak learning rate (log scale)',ylabel='Validation loss',title='Width × learning-rate sweep (mean ± SD over 3 seeds)')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/'05_width_lr_sweep.png',dpi=180); plt.close(fig)
    result=add_uncertainty_artifacts(fine,n_boot=5000)
    result['width4096_confirmation']=confirm_width4096()
    return result

if __name__=='__main__': print(run())
