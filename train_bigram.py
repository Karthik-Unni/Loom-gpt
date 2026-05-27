import torch
from src.tokenizer import CharTokenizer
from src.bigram import BigramLM
from src.dataset import load_data, get_batch

BLOCK_SIZE    = 8
BATCH_SIZE    = 32
MAX_ITERS     = 10_000
EVAL_INTERVAL = 1_000
LR            = 1e-3
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

text = open('data/input.txt').read()
tok  = CharTokenizer(text)
train_data, val_data = load_data('data/input.txt', tok)

model     = BigramLM(tok.vocab_size).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print(f"Training bigram model on {DEVICE}...")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

for step in range(MAX_ITERS + 1):
    xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % EVAL_INTERVAL == 0:
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch(val_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            _, val_loss = model(xv, yv)
        model.train()
        print(f"step {step:5d} | train loss: {loss.item():.4f} | val loss: {val_loss.item():.4f}")

print("\n--- BIGRAM GENERATION (after training) ---")
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
output  = model.generate(context, 300)
print(tok.decode(output[0].tolist()))