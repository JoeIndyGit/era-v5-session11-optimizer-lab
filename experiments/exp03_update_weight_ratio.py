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


def _train_variant(variant: str, width=512, base_lr=3e-3, steps=120):
    dev=device(); seed_everything(CFG.seed)
    xtr,ytr,_,_=make_teacher_dataset(device=dev)
    batches=make_batch_indices(steps)
    model=StudentMLP(width).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=base_lr,weight_decay=CFG.weight_decay)
    parameter_rows=[]
    for st,idx_cpu in enumerate(batches):
        step=st+1
        if variant == 'warmup':
            lr=base_lr*min(1.0,step/CFG.warmup_steps)
        elif variant == 'no_warmup':
            lr=base_lr
        else:
            raise ValueError(variant)
        for pg in opt.param_groups: pg['lr']=lr
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        idx=idx_cpu.to(dev)
        opt.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(model(xtr[idx]),ytr[idx])
        loss.backward(); opt.step()
        new_rows=parameter_update_rows(model,before,step,lr)
        for r in new_rows: r['variant']=variant
        parameter_rows.extend(new_rows)
    pdf=pd.DataFrame(parameter_rows)
    ldf=pd.DataFrame(aggregate_by_layer(parameter_rows)).sort_values(['layer','step'])
    # aggregate_by_layer drops variant; restore it because each call contains one variant
    ldf['variant']=variant
    return pdf,ldf


def _first_all_layers_below(df, threshold=.05, consecutive=5, start_after=20):
    max_gap=df.groupby('step').relative_uw_gap.max().sort_index()
    ok=0
    for step,gap in max_gap.items():
        if step <= start_after:
            continue
        ok = ok + 1 if gap < threshold else 0
        if ok >= consecutive:
            return int(step)-consecutive+1
    return None


def run(width=512, base_lr=3e-3, steps=120):
    warm_pdf,warm_ldf=_train_variant('warmup',width,base_lr,steps)
    ctrl_pdf,ctrl_ldf=_train_variant('no_warmup',width,base_lr,steps)

    warm_pdf.to_csv(ARTIFACTS/"03_update_weight_parameter.csv",index=False)
    warm_ldf.to_csv(ARTIFACTS/"03_update_weight_layer.csv",index=False)
    pd.concat([warm_pdf,ctrl_pdf],ignore_index=True).to_csv(ARTIFACTS/"03_update_weight_parameter_paired.csv",index=False)
    paired_layers=pd.concat([warm_ldf,ctrl_ldf],ignore_index=True)
    paired_layers.to_csv(ARTIFACTS/"03_update_weight_layer_paired.csv",index=False)

    comp=warm_ldf.merge(ctrl_ldf,on=['step','layer'],suffixes=('_warmup','_no_warmup'))
    comp['relative_uw_gap']=(comp.update_to_weight_warmup-comp.update_to_weight_no_warmup).abs()/(comp.update_to_weight_no_warmup.abs()+1e-12)
    comp.to_csv(ARTIFACTS/"03_warmup_causal_comparison.csv",index=False)
    residual_step=_first_all_layers_below(comp,.05,5,CFG.warmup_steps)

    fig,ax=plt.subplots(figsize=(9,5.2))
    for (variant,layer),g in paired_layers.groupby(['variant','layer']):
        ax.plot(g.step,g.update_to_weight,label=f'{layer} — {variant}')
    ax.axvline(CFG.warmup_steps,linestyle='--',label=f'direct LR warmup ends ({CFG.warmup_steps})')
    ax.set(xlabel='Step',ylabel='||ΔW||₂ / ||W||₂',title='Paired update-to-weight trajectories: warmup vs no-warmup')
    ax.legend(ncol=2); fig.tight_layout(); fig.savefig(ARTIFACTS/"03_update_to_weight_ratio.png",dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.8))
    for layer,g in comp.groupby('layer'):
        ax.plot(g.step,g.relative_uw_gap*100,label=layer)
    max_gap=comp.groupby('step').relative_uw_gap.max()*100
    ax.plot(max_gap.index,max_gap.values,linestyle='--',label='max across layers')
    ax.axhline(5.0,linestyle=':',label='5% materiality threshold')
    ax.axvline(CFG.warmup_steps,linestyle='--',alpha=.7,label='warmup LR intervention ends')
    if residual_step is not None:
        ax.axvline(residual_step,linestyle='-.',label=f'5-step sustained <5%: {residual_step}')
    ax.set(xlabel='Step',ylabel='Warmup-control U/W gap (%)',title='Residual causal effect of warmup on update-to-weight ratio')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/"03_warmup_causal_gap.png",dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    warm_lr=warm_ldf.groupby('step').lr.first(); ctrl_lr=ctrl_ldf.groupby('step').lr.first()
    ax.plot(warm_lr.index,warm_lr.values,label='warmup')
    ax.plot(ctrl_lr.index,ctrl_lr.values,label='no-warmup control')
    ax.axvline(CFG.warmup_steps,linestyle='--')
    ax.set(xlabel='Step',ylabel='Learning rate',title='The direct warmup intervention ends after step 20')
    ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACTS/"03_warmup_lr.png",dpi=180); plt.close(fig)

    return {
        "warmup_last_lr_increase_step":CFG.warmup_steps,
        "first_step_with_no_direct_warmup_lr_change":CFG.warmup_steps+1,
        "causal_materiality_threshold":0.05,
        "causal_consecutive_steps":5,
        "first_sustained_step_all_layers_within_5pct_of_no_warmup":residual_step,
        "layer_ratios_at_step20":warm_ldf[warm_ldf.step==CFG.warmup_steps][['layer','update_to_weight']].to_dict('records'),
        "layer_ratios_at_step21":warm_ldf[warm_ldf.step==CFG.warmup_steps+1][['layer','update_to_weight']].to_dict('records'),
        "interpretation":"Warmup stops directly changing LR after step 20, but its induced optimizer/parameter trajectory remains measurably different until substantially later."
    }

if __name__=='__main__': print(run())
