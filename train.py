import os
import argparse
from dataclasses import asdict
import torch
from config import GPTConfig, apply_preset
from src.tokenizer import create_tokenizer
from src.dataset import load_data, get_batch
from src.model import GPT

parser = argparse.ArgumentParser(description='Train a LOOM-GPT transformer.')
parser.add_argument('--data-path')
parser.add_argument('--out-dir')
parser.add_argument('--preset', choices=['tiny', 'laptop', 'single_gpu'], default='laptop')
parser.add_argument('--tokenizer', choices=['byte', 'char'])
parser.add_argument('--max-iters', type=int)
args = parser.parse_args()

cfg = apply_preset(GPTConfig(), args.preset)
if args.data_path:
    cfg.data_path = args.data_path
if args.out_dir:
    cfg.out_dir = args.out_dir
if args.tokenizer:
    cfg.tokenizer = args.tokenizer
if args.max_iters is not None:
    cfg.max_iters = args.max_iters
os.makedirs(cfg.out_dir, exist_ok=True)

# Data
text = open(cfg.data_path, encoding='utf-8').read()
tok  = create_tokenizer(cfg.tokenizer, text)
cfg.vocab_size = tok.vocab_size
train_data, val_data = load_data(cfg.data_path, tok)
if len(val_data) <= cfg.block_size:
    raise ValueError(
        f'Dataset is too small for block size {cfg.block_size}. '
        f'Use more text or choose a smaller preset.'
    )

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
print(f"Vocab size: {cfg.vocab_size} tokens")
print(f"Tokenizer:  {cfg.tokenizer}\n")

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
            torch.save({'model_state': model.state_dict(), 'config': asdict(cfg)}, f"{cfg.out_dir}/best_model.pt")
        print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}{star}")

    if step == cfg.max_iters:
        break

    xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, cfg.device)
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save({'model_state': model.state_dict(), 'config': asdict(cfg)}, f"{cfg.out_dir}/final_model.pt")
print(f"\nSaved: {cfg.out_dir}/best_model.pt  (val loss {best_val:.4f})")
