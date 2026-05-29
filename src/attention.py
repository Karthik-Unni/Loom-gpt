import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """Single head of causal self-attention."""

    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        scale = k.shape[-1] ** -0.5
        wei = q @ k.transpose(-2, -1) * scale  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v     # (B, T, head_size)  