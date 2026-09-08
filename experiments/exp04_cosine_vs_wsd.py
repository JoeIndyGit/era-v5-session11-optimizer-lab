from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from src.config import CFG
from src.data import make_teacher_dataset, make_batch_indices
from src.model import StudentMLP
from src.schedules import cosine_factor, wsd_factor
from src.utils import ARTIFACTS, seed_everything, device


def train_schedule(kind: str, peak_lr: float, width=512, total_steps=300, seed=42, keep_curve=False):
    dev=device(); seed_everything(seed)
    xtr,ytr,xv,yv=make_teacher_dataset(device=dev)
    batches=make_batch_indices(total_steps)
    model=StudentMLP(width).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=peak_lr,weight_decay=CFG.weight_decay)
    rows=[]; checkpoints={}
    for st,idx_cpu in enumerate(batches):
        if kind=='cosine':
            fac=cosine_factor(st,total_steps,CFG.warmup_steps,CFG.min_lr_factor)
        elif kind=='wsd':
            fac=wsd_factor(st,total_steps,CFG.warmup_steps,CFG.wsd_decay_start,CFG.min_lr_factor)
        else:
            raise ValueError(kind)
        lr=peak_lr*fac
        for pg in opt.param_groups: pg['lr']=lr
        idx=idx_cpu.to(dev)
        opt.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(model(xtr[idx]),ytr[idx])
        loss.backward(); opt.step()
        if keep_curve or st+1 in (100,200,300):
            with torch.no_grad():
                val_loss=nn.functional.cross_entropy(model(xv),yv).item()
            rows.append({'schedule':kind,'step':st+1,'peak_lr':peak_lr,'lr':lr,
                         'train_loss':loss.item(),'val_loss':val_loss})
        if st+1 in (100,200,300):
            checkpoints[st+1]={'train_loss':loss.item(),'val_loss':val_loss,'lr':lr}
    return pd.DataFrame(rows), checkpoints


def tune_at_step200(kind: str, candidates):
    records=[]
    for lr in candidates:
        _,ck=train_schedule(kind,lr,total_steps=300,seed=CFG.seed,keep_curve=False)
        r=ck[200]
        records.append({'schedule':kind,'peak_lr':lr,'step200_train_loss':r['train_loss'],'step200_val_loss':r['val_loss']})
    df=pd.DataFrame(records)
    return df, float(df.loc[df.step200_val_loss.idxmin(),'peak_lr'])


def run():
    cosine_grid=[0.0045,0.0050,0.0055,0.0060,0.0065,0.0070]
    wsd_grid=[0.0020,0.0025,0.0030,0.0035,0.0040,0.0045]
    cdf,c_best=tune_at_step200('cosine',cosine_grid)
    wdf,w_best=tune_at_step200('wsd',wsd_grid)
    tune=pd.concat([cdf,wdf],ignore_index=True)
    tune.to_csv(ARTIFACTS/'04_schedule_tuning.csv',index=False)

    ccurve,cck=train_schedule('cosine',c_best,keep_curve=True)
    wcurve,wck=train_schedule('wsd',w_best,keep_curve=True)
    curves=pd.concat([ccurve,wcurve],ignore_index=True)
    curves.to_csv(ARTIFACTS/'04_cosine_vs_wsd_curve.csv',index=False)

    fig,ax=plt.subplots(figsize=(9,4.8))
    for name,g in curves.groupby('schedule'):
        ax.plot(g.step,g.lr,label=f'{name} (peak={g.peak_lr.iloc[0]:.4g})')
    ax.axvline(200,linestyle='--',label='interruption at step 200')
    ax.set(xlabel='Step',ylabel='Learning rate',title='Tuned cosine vs WSD learning-rate schedules')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/'04_cosine_vs_wsd_lr.png',dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5.0))
    for name,g in curves.groupby('schedule'):
        ax.plot(g.step,g.val_loss,label=name)
    ax.axvline(200,linestyle='--',label='interruption at step 200')
    ax.set(xlabel='Step',ylabel='Validation loss',title='Validation loss under independently tuned schedules')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/'04_cosine_vs_wsd_val_loss.png',dpi=180); plt.close(fig)

    keep='cosine' if cck[200]['val_loss'] <= wck[200]['val_loss'] else 'wsd'
    return {
        'best_peak_lr_cosine':c_best,
        'best_peak_lr_wsd':w_best,
        'cosine_step200':cck[200],
        'wsd_step200':wck[200],
        'cosine_step300':cck[300],
        'wsd_step300':wck[300],
        'checkpoint_to_keep_at_step200':keep,
    }

if __name__=='__main__': print(run())
