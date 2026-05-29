from dataclasses import dataclass
import torch

@dataclass
class GPTConfig:
    # Data
    data_path:     str   = 'data/input.txt'
    # Model
    block_size:    int   = 256
    vocab_size:    int   = 65
    n_layer:       int   = 6
    n_head:        int   = 6
    n_embd:        int   = 384
    dropout:       float = 0.2
    # Training
    batch_size:    int   = 64
    learning_rate: float = 3e-4
    max_iters:     int   = 5000
    eval_interval: int   = 500
    eval_iters:    int   = 200
    # System
    device:        str   = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir:       str   = 'out'