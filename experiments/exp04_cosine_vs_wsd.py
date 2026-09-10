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

TUNE_SEEDS=[37,38,39]
EVAL_SEEDS=[41,42,43]
COSINE_LRS=[0.0035,0.0045,0.0055,0.0065,0.0075]
WSD_LRS=[0.0025,0.0035,0.0045,0.0055]
WSD_DECAY_STARTS=[210,240,270]


def train_schedule(kind: str, peak_lr: float, width=512, total_steps=300, seed=42,
                   decay_start=240, eval_every=None):
    """Train one schedule with externally supplied, frozen hyperparameters."""
    dev=device(); seed_everything(seed)
    xtr,ytr,xv,yv=make_teacher_dataset(device=dev)
    batches=make_batch_indices(total_steps)
    model=StudentMLP(width).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=peak_lr,weight_decay=CFG.weight_decay)
    rows=[]; checkpoints={}
    for st,idx_cpu in enumerate(batches):
        step=st+1
        if kind=='cosine':
            fac=cosine_factor(st,total_steps,CFG.warmup_steps,CFG.min_lr_factor)
        elif kind=='wsd':
            fac=wsd_factor(st,total_steps,CFG.warmup_steps,decay_start,CFG.min_lr_factor)
        else:
            raise ValueError(kind)
        lr=peak_lr*fac
        for pg in opt.param_groups: pg['lr']=lr
        idx=idx_cpu.to(dev)
        opt.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(model(xtr[idx]),ytr[idx])
        loss.backward(); opt.step()

        should_eval = step in (200,300) or (eval_every and step % eval_every == 0) or step in (1,20,21)
        if should_eval:
            with torch.no_grad():
                val_loss=nn.functional.cross_entropy(model(xv),yv).item()
            row={
                'schedule':kind,'seed':seed,'step':step,'peak_lr':peak_lr,
                'decay_start':decay_start if kind=='wsd' else None,
                'lr':lr,'train_loss':loss.item(),'val_loss':val_loss,
            }
            rows.append(row)
            if step in (200,300):
                checkpoints[step]=row.copy()
    return pd.DataFrame(rows), checkpoints


def tune_for_planned_horizon():
    """Tune both schedules for the planned step-300 objective on disjoint tuning seeds."""
    records=[]
    for seed in TUNE_SEEDS:
        for lr in COSINE_LRS:
            _,ck=train_schedule('cosine',lr,total_steps=300,seed=seed)
            r=ck[300]
            records.append({
                'schedule':'cosine','peak_lr':lr,'decay_start':None,'tune_seed':seed,
                'step300_train_loss':r['train_loss'],'step300_val_loss':r['val_loss'],
            })
        for decay_start in WSD_DECAY_STARTS:
            for lr in WSD_LRS:
                _,ck=train_schedule('wsd',lr,total_steps=300,seed=seed,decay_start=decay_start)
                r=ck[300]
                records.append({
                    'schedule':'wsd','peak_lr':lr,'decay_start':decay_start,'tune_seed':seed,
                    'step300_train_loss':r['train_loss'],'step300_val_loss':r['val_loss'],
                })

    df=pd.DataFrame(records)
    tuning_summary=(
        df.groupby(['schedule','peak_lr','decay_start'], dropna=False)
          .agg(step300_val_loss_mean=('step300_val_loss','mean'),
               step300_val_loss_std=('step300_val_loss','std'),
               step300_train_loss_mean=('step300_train_loss','mean'))
          .reset_index()
    )
    c=tuning_summary[tuning_summary.schedule=='cosine'].loc[
        tuning_summary[tuning_summary.schedule=='cosine'].step300_val_loss_mean.idxmin()
    ]
    w=tuning_summary[tuning_summary.schedule=='wsd'].loc[
        tuning_summary[tuning_summary.schedule=='wsd'].step300_val_loss_mean.idxmin()
    ]
    selected={
        'cosine':{'peak_lr':float(c.peak_lr),'decay_start':None},
        'wsd':{'peak_lr':float(w.peak_lr),'decay_start':int(w.decay_start)},
    }
    return df,tuning_summary,selected


def run():
    tuning,tuning_summary,selected=tune_for_planned_horizon()
    tuning.to_csv(ARTIFACTS/'04_schedule_tuning.csv',index=False)
    tuning_summary.to_csv(ARTIFACTS/'04_schedule_tuning_summary.csv',index=False)

    eval_rows=[]
    for seed in EVAL_SEEDS:
        for kind in ('cosine','wsd'):
            hp=selected[kind]
            curve,_=train_schedule(
                kind,hp['peak_lr'],total_steps=300,seed=seed,
                decay_start=hp['decay_start'] or CFG.wsd_decay_start,eval_every=10,
            )
            eval_rows.append(curve)
    curves=pd.concat(eval_rows,ignore_index=True)
    curves.to_csv(ARTIFACTS/'04_schedule_multiseed_curve.csv',index=False)
    curves.to_csv(ARTIFACTS/'04_cosine_vs_wsd_curve.csv',index=False)

    summary=(
        curves[curves.step.isin([200,300])]
        .groupby(['schedule','step'])
        .agg(train_loss_mean=('train_loss','mean'),train_loss_std=('train_loss','std'),
             val_loss_mean=('val_loss','mean'),val_loss_std=('val_loss','std'))
        .reset_index()
    )
    summary.to_csv(ARTIFACTS/'04_frozen_multiseed_summary.csv',index=False)

    paired=(
        curves[curves.step.isin([200,300])]
        .pivot_table(index=['seed','step'],columns='schedule',values='val_loss')
        .reset_index()
    )
    paired['wsd_minus_cosine']=paired['wsd']-paired['cosine']
    paired.to_csv(ARTIFACTS/'04_paired_seed_differences.csv',index=False)

    fig,ax=plt.subplots(figsize=(9,5.0))
    for name,g in tuning_summary.groupby('schedule'):
        if name=='cosine':
            gg=g.sort_values('peak_lr')
            ax.errorbar(gg.peak_lr,gg.step300_val_loss_mean,yerr=gg.step300_val_loss_std,
                        marker='o',capsize=3,label='cosine')
        else:
            for decay,gg in g.groupby('decay_start'):
                gg=gg.sort_values('peak_lr')
                ax.errorbar(gg.peak_lr,gg.step300_val_loss_mean,yerr=gg.step300_val_loss_std,
                            marker='o',capsize=3,label=f'WSD decay@{int(decay)}')
    ax.set_xscale('log')
    ax.set(xlabel='Peak LR',ylabel='Validation loss at planned step 300',
           title='Tune both schedules for the planned horizon before interruption analysis')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/'04_schedule_tuning.png',dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.8))
    for name,g in curves.groupby('schedule'):
        g0=g[g.seed==EVAL_SEEDS[0]].sort_values('step')
        ax.plot(g0.step,g0.lr,label=f"{name} (peak={g0.peak_lr.iloc[0]:.4g})")
    ax.axvline(200,linestyle='--',label='unexpected interruption at step 200')
    ax.set(xlabel='Step',ylabel='Learning rate',title='Frozen cosine vs WSD schedules (planned horizon = 300)')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/'04_cosine_vs_wsd_lr.png',dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5.0))
    agg=curves.groupby(['schedule','step']).agg(mean=('val_loss','mean'),std=('val_loss','std')).reset_index()
    for name,g in agg.groupby('schedule'):
        g=g.sort_values('step')
        ax.plot(g.step,g['mean'],label=name)
        ax.fill_between(g.step,g['mean']-g['std'].fillna(0),g['mean']+g['std'].fillna(0),alpha=.18)
    ax.axvline(200,linestyle='--',label='unexpected interruption at step 200')
    ax.set(xlabel='Step',ylabel='Validation loss',title='Frozen hyperparameters: validation loss mean ± SD over 3 seeds')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/'04_cosine_vs_wsd_val_loss.png',dpi=180); plt.close(fig)

    s200=summary[summary.step==200].set_index('schedule')
    s300=summary[summary.step==300].set_index('schedule')
    keep=str(s200.val_loss_mean.idxmin())
    return {
        'tuning_objective':'validation loss at the originally planned 300-step horizon',
        'tuning_seeds':TUNE_SEEDS,
        'evaluation_seeds':EVAL_SEEDS,
        'selected_cosine_peak_lr':selected['cosine']['peak_lr'],
        'selected_wsd_peak_lr':selected['wsd']['peak_lr'],
        'selected_wsd_decay_start':selected['wsd']['decay_start'],
        'cosine_step200_val_mean':float(s200.loc['cosine'].val_loss_mean),
        'cosine_step200_val_std':float(s200.loc['cosine'].val_loss_std),
        'wsd_step200_val_mean':float(s200.loc['wsd'].val_loss_mean),
        'wsd_step200_val_std':float(s200.loc['wsd'].val_loss_std),
        'cosine_step300_val_mean':float(s300.loc['cosine'].val_loss_mean),
        'cosine_step300_val_std':float(s300.loc['cosine'].val_loss_std),
        'wsd_step300_val_mean':float(s300.loc['wsd'].val_loss_mean),
        'wsd_step300_val_std':float(s300.loc['wsd'].val_loss_std),
        'checkpoint_to_keep_at_step200':keep,
        'fairness_note':'Step 200 is never used for hyperparameter selection; both schedules are tuned at the planned step-300 horizon on seeds disjoint from final evaluation.',
    }


if __name__=='__main__':
    print(run())
