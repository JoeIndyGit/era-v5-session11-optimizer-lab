from __future__ import annotations
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from src.config import CFG
from src.schedules import cosine_factor,wsd_factor
from src.tiny_transformer import TinyTransformerLM,make_synthetic_language
from src.utils import ARTIFACTS,seed_everything,device

TUNE_SEED=40
EVAL_SEEDS=[41,42,43]
COSINE_LRS=[0.10,0.20,0.40,0.80]
WSD_LRS=[0.10,0.20,0.40,0.80]
DECAY_STARTS=[210,240]


def _dataset(dev):
    x=make_synthetic_language()
    return x[:2048].to(dev),x[2048:].to(dev)


def _batches(steps,batch_size=32,seed=1234):
    g=torch.Generator(device='cpu').manual_seed(seed)
    return [torch.randint(0,2048,(batch_size,),generator=g) for _ in range(steps)]


def _loss(model,seqs):
    inp=seqs[:,:-1]; target=seqs[:,1:]
    logits=model(inp)
    return nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1))


def train(kind,peak_lr,seed=42,decay_start=210,total_steps=300,eval_every=None):
    warnings.filterwarnings('ignore',message='enable_nested_tensor')
    dev=device();seed_everything(seed)
    train_x,val_x=_dataset(dev); batches=_batches(total_steps)
    model=TinyTransformerLM().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=peak_lr,weight_decay=0.01)
    rows=[]; checkpoints={}
    for st,idx_cpu in enumerate(batches):
        step=st+1
        fac=cosine_factor(st,total_steps,20,0.05) if kind=='cosine' else wsd_factor(st,total_steps,20,decay_start,0.05)
        lr=peak_lr*fac
        for pg in opt.param_groups: pg['lr']=lr
        idx=idx_cpu.to(dev);opt.zero_grad(set_to_none=True)
        loss=_loss(model,train_x[idx]);loss.backward();opt.step()
        should_eval=step in (200,300) or (eval_every and step%eval_every==0) or step in (1,20,21)
        if should_eval:
            with torch.no_grad(): val_loss=float(_loss(model,val_x[:128]).item())
            row={'schedule':kind,'seed':seed,'step':step,'peak_lr':peak_lr,'decay_start':decay_start if kind=='wsd' else None,
                 'lr':lr,'train_loss':float(loss.item()),'val_loss':val_loss}
            rows.append(row)
            if step in (200,300): checkpoints[step]=row.copy()
    return pd.DataFrame(rows),checkpoints


def run():
    tuning=[]
    for lr in COSINE_LRS:
        _,ck=train('cosine',lr,seed=TUNE_SEED)
        tuning.append({'schedule':'cosine','peak_lr':lr,'decay_start':None,'step300_val_loss':ck[300]['val_loss']})
    for decay in DECAY_STARTS:
        for lr in WSD_LRS:
            _,ck=train('wsd',lr,seed=TUNE_SEED,decay_start=decay)
            tuning.append({'schedule':'wsd','peak_lr':lr,'decay_start':decay,'step300_val_loss':ck[300]['val_loss']})
    tune=pd.DataFrame(tuning);tune.to_csv(ARTIFACTS/'06_transformer_schedule_tuning.csv',index=False)
    c=tune[tune.schedule=='cosine'].loc[tune[tune.schedule=='cosine'].step300_val_loss.idxmin()]
    w=tune[tune.schedule=='wsd'].loc[tune[tune.schedule=='wsd'].step300_val_loss.idxmin()]
    selected={'cosine':(float(c.peak_lr),210),'wsd':(float(w.peak_lr),int(w.decay_start))}

    rows=[]
    for seed in EVAL_SEEDS:
        for kind in ('cosine','wsd'):
            lr,decay=selected[kind]
            curve,_=train(kind,lr,seed=seed,decay_start=decay,eval_every=50)
            rows.append(curve)
    curves=pd.concat(rows,ignore_index=True);curves.to_csv(ARTIFACTS/'06_transformer_multiseed_curve.csv',index=False)
    summary=(curves[curves.step.isin([200,300])].groupby(['schedule','step'])
             .agg(val_loss_mean=('val_loss','mean'),val_loss_std=('val_loss','std'),train_loss_mean=('train_loss','mean'))
             .reset_index())
    summary.to_csv(ARTIFACTS/'06_transformer_summary.csv',index=False)

    fig,ax=plt.subplots(figsize=(9,5.0))
    agg=curves.groupby(['schedule','step']).agg(mean=('val_loss','mean'),std=('val_loss','std')).reset_index()
    for name,g in agg.groupby('schedule'):
        ax.plot(g.step,g['mean'],label=name)
        ax.fill_between(g.step,g['mean']-g['std'].fillna(0),g['mean']+g['std'].fillna(0),alpha=.18)
    ax.axvline(200,linestyle='--',label='interruption at step 200')
    ax.set(xlabel='Step',ylabel='Token validation loss',title='Causal Transformer schedule replication (mean ± SD, 3 seeds)')
    ax.legend();fig.tight_layout();fig.savefig(ARTIFACTS/'06_transformer_schedule_replication.png',dpi=180);plt.close(fig)

    s200=summary[summary.step==200].set_index('schedule');s300=summary[summary.step==300].set_index('schedule')
    return {
        'purpose':'External-validity replication: repeat the schedule protocol on a causal Transformer rather than the MLP benchmark.',
        'selected_cosine_peak_lr':selected['cosine'][0],
        'selected_wsd_peak_lr':selected['wsd'][0],
        'selected_wsd_decay_start':selected['wsd'][1],
        'cosine_step200_val_mean':float(s200.loc['cosine'].val_loss_mean),'cosine_step200_val_std':float(s200.loc['cosine'].val_loss_std),
        'wsd_step200_val_mean':float(s200.loc['wsd'].val_loss_mean),'wsd_step200_val_std':float(s200.loc['wsd'].val_loss_std),
        'cosine_step300_val_mean':float(s300.loc['cosine'].val_loss_mean),'wsd_step300_val_mean':float(s300.loc['wsd'].val_loss_mean),
        'step200_winner':str(s200.val_loss_mean.idxmin()),
        'note':'This replication checks whether the schedule conclusion survives an architecture change.'
    }

if __name__=='__main__': print(run())
