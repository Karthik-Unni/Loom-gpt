from dataclasses import dataclass
import torch

@dataclass
class GPTConfig:
    # Data
    data_path:     str   = 'data/input.txt'
    tokenizer:     str   = 'byte'
    # Model
    block_size:    int   = 128
    vocab_size:    int   = 65
    n_layer:       int   = 4
    n_head:        int   = 4
    n_embd:        int   = 128
    dropout:       float = 0.2
    # Training
    batch_size:    int   = 16
    learning_rate: float = 3e-4
    max_iters:     int   = 1000
    eval_interval: int   = 100
    eval_iters:    int   = 10
    # System
    device:        str   = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir:       str   = 'out'


PRESETS = {
    'tiny': {
        'block_size': 64, 'n_layer': 2, 'n_head': 2, 'n_embd': 64,
        'batch_size': 8, 'max_iters': 300, 'eval_interval': 50, 'eval_iters': 5,
    },
    'laptop': {
        'block_size': 128, 'n_layer': 4, 'n_head': 4, 'n_embd': 128,
        'batch_size': 16, 'max_iters': 1000, 'eval_interval': 100, 'eval_iters': 10,
    },
    'single_gpu': {
        'block_size': 256, 'n_layer': 6, 'n_head': 6, 'n_embd': 384,
        'batch_size': 64, 'max_iters': 5000, 'eval_interval': 500, 'eval_iters': 100,
    },
}


def apply_preset(cfg: GPTConfig, name: str) -> GPTConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Choose from: {', '.join(PRESETS)}")
    for key, value in PRESETS[name].items():
        setattr(cfg, key, value)
    return cfg
