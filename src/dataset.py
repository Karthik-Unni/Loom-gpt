import torch


# Load raw text, tokenize it, and split into train/validation tensors
def load_data(filepath: str, tokenizer, train_ratio: float = 0.9):

    # Read entire corpus into memory
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Convert characters → integer token ids
    # Example: "Hello" → [20, 43, 50, 50, 53]
    data = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long
    )

    # Split dataset into train and validation
    # Default:
    # 90% → training
    # 10% → validation
    n = int(train_ratio * len(data))

    return data[:n], data[n:]


# Sample random mini-batches for training
def get_batch(
    data,
    block_size: int,
    batch_size: int,
    device: str = "cpu"
):

    # Random starting positions
    # Each index becomes one training example
    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    # Input sequence
    # Example:
    # x = [H,e,l,l]
    x = torch.stack([
        data[i:i+block_size]
        for i in ix
    ])

    # Target sequence (shifted right)
    # Example:
    # y = [e,l,l,o]
    # Model learns next-token prediction
    y = torch.stack([
        data[i+1:i+block_size+1]
        for i in ix
    ])

    # Move tensors to CPU/GPU
    return x.to(device), y.to(device)

'''
For testing this you can try out 
python -c " 
from src.tokenizer import CharTokenizer
from src.dataset import load_data, get_batch

text = open('data/input.txt').read()
tok = CharTokenizer(text)
train, val = load_data('data/input.txt', tok)

print(f'Train tokens: {len(train):,}')
print(f'Val tokens:   {len(val):,}')

xb, yb = get_batch(train, block_size=8, batch_size=4)
print(f'\nBatch x shape: {xb.shape}  (batch, time)')
print(f'Batch y shape: {yb.shape}  (batch, time)')
print()
print('x[0] (encoded):', xb[0].tolist())
print('y[0] (encoded):', yb[0].tolist())
print('x[0] (text):   ', repr(tok.decode(xb[0].tolist())))
print('y[0] (text):   ', repr(tok.decode(yb[0].tolist())))
print()
print('Notice: y is x shifted 1 position to the right.')
print('At every position, predict the next character.')
"



The expected outcome of this will be 

Train tokens: 1,003,854
Val tokens:   111,540

Batch x shape: torch.Size([4, 8])  (batch, time)
Batch y shape: torch.Size([4, 8])  (batch, time)

x[0] (encoded): [43, 1, 39, 1, 54, 56, 43, 41]
y[0] (encoded): [1, 39, 1, 54, 56, 43, 41, 43]
x[0] (text):    'e a prec'
y[0] (text):    ' a prece'

Notice: y is x shifted 1 position to the right.
At every position, predict the next character.
'''
