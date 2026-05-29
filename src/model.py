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
    
class Block(nn.Module):
    """One transformer block: attention then feedforward, with residuals."""

    def __init__(self, n_embd: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa  = MultiHeadAttention(n_heads, head_size, n_embd, block_size, dropout)
        self.ffn = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # attend + residual
        x = x + self.ffn(self.ln2(x))  # compute + residual
        return x
    
class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_heads, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb    = nn.Embedding(vocab_size, n_embd)
        self.pos_emb    = nn.Embedding(block_size, n_embd)
        self.blocks     = nn.Sequential(*[
            Block(n_embd, n_heads, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying — token embedding and lm_head share weights (GPT-2 trick)
        self.tok_emb.weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T    = idx.shape
        device  = idx.device
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x       = tok_emb + pos_emb
        x       = self.blocks(x)
        x       = self.ln_f(x)
        logits  = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, idx_next], dim=1)
        return idx