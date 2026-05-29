import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attention import MultiHeadAttention

class FeedForward(nn.Module):
    """Position-wise MLP: each token processed independently."""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)