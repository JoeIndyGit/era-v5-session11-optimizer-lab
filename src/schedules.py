from __future__ import annotations
import math


def linear_warmup(step: int, warmup_steps: int) -> float:
    s = step + 1
    return min(1.0, s / max(1, warmup_steps))


def cosine_factor(step: int, total_steps: int, warmup_steps: int, min_lr_factor: float = 0.05) -> float:
    s = step + 1
    if s <= warmup_steps:
        return s / warmup_steps
    p = (s - warmup_steps) / max(1, total_steps - warmup_steps)
    p = min(max(p, 0.0), 1.0)
    return min_lr_factor + (1 - min_lr_factor) * 0.5 * (1 + math.cos(math.pi * p))


def wsd_factor(step: int, total_steps: int, warmup_steps: int, decay_start: int, min_lr_factor: float = 0.05) -> float:
    s = step + 1
    if s <= warmup_steps:
        return s / warmup_steps
    if s <= decay_start:
        return 1.0
    p = (s - decay_start) / max(1, total_steps - decay_start)
    p = min(max(p, 0.0), 1.0)
    return min_lr_factor + (1 - min_lr_factor) * (1 - p)
