import argparse
import os
from dataclasses import asdict

import torch

from config import GPTConfig, apply_preset
from src.dataset import get_batch, load_data
from src.model import GPT
from src.tokenizer import create_tokenizer
from src.training import EarlyStopping, HistoryLogger


parser = argparse.ArgumentParser(description='Train a LOOM-GPT transformer.')
parser.add_argument('--data-path')
parser.add_argument('--out-dir')
parser.add_argument('--preset', choices=['tiny', 'laptop', 'single_gpu'], default='laptop')
parser.add_argument('--tokenizer', choices=['byte', 'char'])
parser.add_argument('--max-iters', type=int)
parser.add_argument('--early-stopping', type=int, default=8, metavar='EVALS')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--resume', help='Resume model and optimizer state from a checkpoint.')
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

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Data
text = open(cfg.data_path, encoding='utf-8').read()
tok = create_tokenizer(cfg.tokenizer, text)
cfg.vocab_size = tok.vocab_size
train_data, val_data = load_data(cfg.data_path, tok)
if len(val_data) <= cfg.block_size:
    raise ValueError(
        f'Dataset is too small for block size {cfg.block_size}. '
        f'Use more text or choose a smaller preset.'
    )
if len(train_data) < 100_000:
    print(f"Warning: small dataset ({len(train_data):,} training tokens). Add more clean text for better output.\n")

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
start_step = 0
stopper = EarlyStopping(args.early_stopping)

if args.resume:
    checkpoint = torch.load(args.resume, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state'])
    if 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_step = checkpoint.get('step', 0) + 1
    stopper.best_loss = checkpoint.get('best_val', float('inf'))
    stopper.best_step = checkpoint.get('best_step', 0)
    print(f"Resuming:   {args.resume} from step {start_step}\n")

history = HistoryLogger(f"{cfg.out_dir}/history.csv", append=bool(args.resume))


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


def save_checkpoint(path: str, step: int) -> None:
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'config': asdict(cfg),
        'step': step,
        'best_val': stopper.best_loss,
        'best_step': stopper.best_step,
        'seed': args.seed,
    }, path)


for step in range(start_step, cfg.max_iters + 1):
    if step % cfg.eval_interval == 0:
        losses = estimate_loss()
        history.append(step, losses['train'], losses['val'])
        improved = stopper.update(step, losses['val'])
        star = " <- best" if improved else ""
        if improved:
            save_checkpoint(f"{cfg.out_dir}/best_model.pt", step)
        print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}{star}")
        if stopper.should_stop:
            print(f"Early stopping at step {step}. Best step: {stopper.best_step}")
            break

    if step == cfg.max_iters:
        break

    xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, cfg.device)
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

save_checkpoint(f"{cfg.out_dir}/final_model.pt", step)
print(f"\nSaved: {cfg.out_dir}/best_model.pt  (step {stopper.best_step}, val loss {stopper.best_loss:.4f})")
print(f"History: {cfg.out_dir}/history.csv")
