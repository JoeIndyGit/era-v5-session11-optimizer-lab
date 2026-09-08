from __future__ import annotations
from torch import nn
from .config import CFG


class StudentMLP(nn.Module):
    """Small width-controlled student network.

    Only hidden width changes in width-scaling experiments; input/output task stays fixed.
    """
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(CFG.input_dim, width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width, CFG.n_classes)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
