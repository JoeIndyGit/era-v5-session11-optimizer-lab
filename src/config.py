from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    input_dim: int = 64
    teacher_hidden: int = 128
    n_classes: int = 20
    train_size: int = 10_000
    val_size: int = 2_000
    batch_size: int = 128
    weight_decay: float = 1e-2
    warmup_steps: int = 20
    total_steps: int = 300
    wsd_decay_start: int = 240
    min_lr_factor: float = 0.05

CFG = ExperimentConfig()
