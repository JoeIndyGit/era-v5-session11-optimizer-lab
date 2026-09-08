from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from src.config import CFG
from src.data import make_teacher_dataset, make_batch_indices
from src.model import StudentMLP
from src.metrics import parameter_update_rows, aggregate_by_layer
from src.utils import ARTIFACTS, seed_everything, device


def run(width=512, base_lr=3e-3, steps=80):
    dev=device(); seed_everything(CFG.seed)
    xtr,ytr,_,_=make_teacher_dataset(device=dev)
    batches=make_batch_indices(steps)
    model=StudentMLP(width).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=base_lr,weight_decay=CFG.weight_decay)
    parameter_rows=[]
    for st,idx_cpu in enumerate(batches):
        step=st+1; lr=base_lr*min(1.0,step/CFG.warmup_steps)
        for pg in opt.param_groups: pg['lr']=lr
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        idx=idx_cpu.to(dev)
        opt.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(model(xtr[idx]),ytr[idx])
        loss.backward(); opt.step()
        parameter_rows.extend(parameter_update_rows(model,before,step,lr))
    pdf=pd.DataFrame(parameter_rows)
    ldf=pd.DataFrame(aggregate_by_layer(parameter_rows)).sort_values(['layer','step'])
    pdf.to_csv(ARTIFACTS/"03_update_weight_parameter.csv",index=False)
    ldf.to_csv(ARTIFACTS/"03_update_weight_layer.csv",index=False)

    fig,ax=plt.subplots(figsize=(9,5.2))
    for layer,g in ldf.groupby('layer'):
        ax.plot(g.step,g.update_to_weight,label=layer)
    ax.axvline(CFG.warmup_steps,linestyle='--',label=f'warmup ends ({CFG.warmup_steps})')
    ax.set(xlabel='Step',ylabel='||ΔW||₂ / ||W||₂',title='Per-layer update-to-weight ratio during and after warmup')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/"03_update_to_weight_ratio.png",dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    lr_by_step=ldf.groupby('step').lr.first()
    ax.plot(lr_by_step.index,lr_by_step.values)
    ax.axvline(CFG.warmup_steps,linestyle='--')
    ax.set(xlabel='Step',ylabel='Learning rate',title='Linear warmup followed by a flat LR')
    fig.tight_layout(); fig.savefig(ARTIFACTS/"03_warmup_lr.png",dpi=180); plt.close(fig)

    return {
        "warmup_last_lr_increase_step":CFG.warmup_steps,
        "first_step_with_no_warmup_lr_change":CFG.warmup_steps+1,
        "layer_ratios_at_step20":ldf[ldf.step==CFG.warmup_steps][['layer','update_to_weight']].to_dict('records'),
        "layer_ratios_at_step21":ldf[ldf.step==CFG.warmup_steps+1][['layer','update_to_weight']].to_dict('records'),
    }

if __name__=='__main__': print(run())
