from __future__ import annotations
from collections import defaultdict
import math


def parameter_update_rows(model, before, step: int, lr: float, eps: float = 1e-12):
    rows = []
    for name, p in model.named_parameters():
        w0 = before[name]
        delta = p.detach() - w0
        wn = w0.norm().item()
        un = delta.norm().item()
        rows.append({
            "step": step,
            "parameter": name,
            "layer": name.split(".")[0],
            "lr": lr,
            "weight_norm": wn,
            "update_norm": un,
            "update_to_weight": un / (wn + eps),
        })
    return rows


def aggregate_by_layer(rows, eps: float = 1e-12):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["step"], r["layer"], r["lr"])].append(r)
    out = []
    for (step, layer, lr), rs in groups.items():
        wn = math.sqrt(sum(r["weight_norm"] ** 2 for r in rs))
        un = math.sqrt(sum(r["update_norm"] ** 2 for r in rs))
        out.append({
            "step": step, "layer": layer, "lr": lr,
            "weight_norm": wn, "update_norm": un,
            "update_to_weight": un / (wn + eps),
        })
    return out
