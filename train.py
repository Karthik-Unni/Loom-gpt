import os
import torch
from config import GPTConfig
from src.tokenizer import CharTokenizer
from src.dataset import load_data, get_batch
from src.model import GPT

cfg = GPTConfig()
os.makedirs(cfg.out_dir, exist_ok=True)

# Data
text = open(cfg.data_path).read()
tok  = CharTokenizer(text)
cfg.vocab_size = tok.vocab_size
train_data, val_data = load_data(cfg.data_path, tok)

# Model
model = GPT(
    vocab_size=cfg.vocab_size,
    n_embd=cfg.n_embd,
    n_heads=cfg.n_head,
    n_layer=cfg.n_layer,
    block_size=cfg.block_size,
    dropout=cfg.dropout,
).to(cfg.device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model:      GPT  ({cfg.n_layer}L / {cfg.n_head}H / {cfg.n_embd}D)")
print(f"Parameters: {n_params:,}")
print(f"Device:     {cfg.device}")
print(f"Block size: {cfg.block_size} tokens")
print(f"Vocab size: {cfg.vocab_size} chars\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = [
            model(*get_batch(data, cfg.block_size, cfg.batch_size, cfg.device))[1].item()
            for _ in range(cfg.eval_iters)
        ]
        results[split] = sum(losses) / len(losses)
    model.train()
    return results

best_val = float('inf')

for step in range(cfg.max_iters + 1):
    if step % cfg.eval_interval == 0:
        losses = estimate_loss()
        star = " ← best" if losses['val'] < best_val else ""
        if losses['val'] < best_val:
            best_val = losses['val']
            torch.save(model.state_dict(), f"{cfg.out_dir}/best_model.pt")
        print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}{star}")

    if step == cfg.max_iters:
        break

    xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, cfg.device)
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), f"{cfg.out_dir}/final_model.pt")
print(f"\nSaved: {cfg.out_dir}/best_model.pt  (val loss {best_val:.4f})")