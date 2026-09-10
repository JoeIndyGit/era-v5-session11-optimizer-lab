from __future__ import annotations
import math
import torch
from torch import nn

VOCAB_SIZE = 32
SEQ_LEN = 16
N_SEQUENCES = 2560


def make_synthetic_language(n_sequences: int = N_SEQUENCES, seq_len: int = SEQ_LEN,
                            vocab_size: int = VOCAB_SIZE, seed: int = 2026) -> torch.Tensor:
    """Create a deterministic causal-language dataset with a four-token-back dependency.

    The first four tokens are sampled uniformly. Every later token follows
    x_t = (x_{t-1} + 1) mod vocab_size, so a model must use sequence context
    rather than solve a static classification problem.
    """
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.empty((n_sequences, seq_len), dtype=torch.long)
    x[:, 0] = torch.randint(0, vocab_size, (n_sequences,), generator=g)
    for t in range(1, seq_len):
        x[:, t] = (x[:, t - 1] + 1) % vocab_size
    return x


class TinyTransformerLM(nn.Module):
    """Small causal Transformer language model used as an architecture replication."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, seq_len: int = SEQ_LEN,
                 d_model: int = 16, n_heads: int = 2, n_layers: int = 1,
                 dim_feedforward: int = 64, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        b, t = tokens.shape
        if t > self.seq_len:
            raise ValueError(f"sequence length {t} exceeds configured maximum {self.seq_len}")
        pos = torch.arange(t, device=tokens.device)
        h = self.token_emb(tokens) * math.sqrt(self.token_emb.embedding_dim)
        h = h + self.pos_emb(pos)[None, :, :]
        causal_mask = torch.triu(
            torch.full((t, t), float("-inf"), device=tokens.device), diagonal=1
        )
        h = self.blocks(h, mask=causal_mask)
        h = self.norm(h)
        return self.lm_head(h)
