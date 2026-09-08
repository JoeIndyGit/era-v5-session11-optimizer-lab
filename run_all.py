from __future__ import annotations
import json
from pathlib import Path
from experiments.exp01_adam_by_hand import run as run1
from experiments.exp02_bias_correction import run as run2
from experiments.exp03_update_weight_ratio import run as run3
from experiments.exp04_cosine_vs_wsd import run as run4
from experiments.exp05_width_lr_sweep import run as run5
from src.utils import ARTIFACTS


def main():
    results={}
    for name,fn in [
        ('adam_by_hand',run1),('bias_correction',run2),('update_to_weight',run3),
        ('cosine_vs_wsd',run4),('width_lr_sweep',run5)]:
        print(f'Running {name}...')
        results[name]=fn()
    with (ARTIFACTS/'summary.json').open('w',encoding='utf-8') as f:
        json.dump(results,f,indent=2)
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
