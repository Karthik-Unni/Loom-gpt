import torch
import argparse
from config import GPTConfig
from src.tokenizer import CharTokenizer
from src.model import GPT

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',   default='out/best_model.pt')
parser.add_argument('--tokens',       type=int,   default=500)
parser.add_argument('--temperature',  type=float, default=0.8)
parser.add_argument('--top_k',        type=int,   default=40)
parser.add_argument('--prompt',       type=str,   default='')
args = parser.parse_args()

cfg  = GPTConfig()
cfg.dropout = 0.0   # no dropout at inference

text = open(cfg.data_path).read()
tok  = CharTokenizer(text)
cfg.vocab_size = tok.vocab_size

model = GPT(
    vocab_size=cfg.vocab_size,
    n_embd=cfg.n_embd,
    n_heads=cfg.n_head,
    n_layer=cfg.n_layer,
    block_size=cfg.block_size,
    dropout=cfg.dropout,
).to(cfg.device)

model.load_state_dict(torch.load(args.checkpoint, map_location=cfg.device))
model.eval()

if args.prompt:
    context = torch.tensor(tok.encode(args.prompt), dtype=torch.long)[None].to(cfg.device)
else:
    context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)

print(f"--- loom-gpt | temp={args.temperature} | top_k={args.top_k} ---\n")
if args.prompt:
    print(args.prompt, end='', flush=True)

output = model.generate(context, args.tokens, temperature=args.temperature, top_k=args.top_k)
generated_text = tok.decode(output[0].tolist())

if args.prompt:
    print(generated_text[len(args.prompt):])
else:
    print(generated_text)