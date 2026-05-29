# 🧵 Loom-GPT

A GPT-style language model built from scratch in PyTorch by implementing the core components of the Transformer architecture step-by-step.

---

## Features

### Character-Level Tokenization

Loom-GPT converts raw text into character tokens and learns language patterns directly from the data without relying on any pre-trained tokenizer. This helps illustrate the complete language modeling pipeline from the ground up.

### Self-Attention from Scratch

The attention mechanism is implemented manually using Query, Key, and Value projections. This allows the model to learn which previous tokens are most relevant when predicting the next character.

### Multi-Head Attention

Multiple attention heads operate in parallel, enabling the model to capture different contextual relationships within the text. Their outputs are combined to build richer token representations.

### Transformer Architecture

The model is built using stacked Transformer blocks consisting of multi-head self-attention, feed-forward networks, residual connections, and layer normalization, closely following modern GPT architectures.

### Autoregressive Text Generation

Loom-GPT generates text one token at a time by predicting the next character based on previously generated context. Temperature and top-k sampling are supported for controllable generation.

### Training & Checkpointing

The project includes a complete training pipeline with validation loss monitoring, model checkpointing, and configurable hyperparameters for experimentation and reproducibility.

### Modular Design

Each component of the architecture is separated into dedicated modules, making the codebase easier to understand, extend, and experiment with while learning Transformer internals.


---

## Architecture

```text
Input Text
    ↓
Character Tokenizer
    ↓
Token + Position Embeddings
    ↓
Transformer Blocks
    ├── Multi-Head Self-Attention
    ├── Feed Forward Network
    └── Residual Connections
    ↓
LayerNorm
    ↓
Language Modeling Head
    ↓
Next Character Prediction
```

---

## Model Configuration

| Component       | Value  |
| --------------- | ------ |
| Layers          | 6      |
| Attention Heads | 6      |
| Embedding Size  | 384    |
| Context Length  | 256    |
| Parameters      | ~10.8M |

---

## Training Results

| Model    | Validation Loss |
| -------- | --------------- |
| Full GPT | 1.58            |

The model learns Shakespeare-style structure, speaker formatting, punctuation patterns, and character-level language generation from raw text.

---

## Example Generation

```text
KING RICHARD III:

What means this? Speak, thou fearful man:
Is it the morning that hath brought thee here,
Or art thou come to mock us with thy tongue?
```

---

## Project Structure

```text
loom-gpt/
├── data/
├── notebooks/
├── src/
│   ├── attention.py
│   ├── dataset.py
│   ├── model.py
│   └── tokenizer.py
├── train.py
├── generate.py
├── config.py
└── README.md
```

---

## Run Training

```bash
python train.py
```

## Generate Text

```bash
python generate.py
```

## Acknowledgements

* Andrej Karpathy — Let's Build GPT
* Attention Is All You Need (Vaswani et al., 2017)

---

Built as a learning-focused implementation to understand how GPT models work internally, from embeddings to text generation.
