# 🧵 LOOM-GPT
### Building GPT from Scratch — One Layer at a Time

> Weaving language from matrices, attention, and next-token prediction.

A from-scratch implementation of GPT following Andrej Karpathy's  
**"Let's Build GPT: from scratch, in code, spelled out"** — but built as an engineering project with incremental commits, experiments, notes, and upgrades.

---

## Current Progress → Commit 6 ✅

Progress:
- [x] Project setup
- [x] Tiny Shakespeare dataset
- [x] Character-level tokenizer
- [x] Dataset pipeline + batch sampling
- [x] Bigram Language Model
- [x] Bigram training loop
- [x] Attention intuition (averaging trick)
- [ ] Self-attention
- [ ] Multi-head attention
- [ ] Transformer block
- [ ] Full GPT
- [ ] Training + generation
- [ ] Upgrades

---

# What This Project Is

LOOM-GPT is an attempt to understand GPT from first principles.

Instead of using APIs or pretrained models, this project builds every component manually:

```text
Raw Text
↓
Tokenizer
↓
Tokens
↓
Dataset
↓
Language Model
↓
Training
↓
Attention
↓
Transformer
↓
GPT
```

Current stage:
We moved from predicting text using simple next-character statistics → toward learning contextual relationships using attention.

---

# Implemented So Far

## 1. Character Tokenizer

Converts text into integers.

Example:

```text
"Hello"

↓

[20,43,50,50,53]
```

Concept learned:
- Vocabulary
- Encoding / Decoding
- Text representation

---

## 2. Dataset Pipeline

Creates training pairs.

Example:

```text
x = Hell
y = ello
```

Model learns:

```text
H → e
e → l
l → l
l → o
```

Concept learned:
- Block size
- Batch sampling
- Next-token prediction

---

## 3. Bigram Language Model

First neural language model.

Idea:

```text
Current Character
↓

Predict Next Character
```

Built with:

```python
nn.Embedding
Cross Entropy
Softmax
Sampling
```

Concept learned:
- Embeddings
- Forward pass
- Loss
- Generation

---

## 4. Training Loop

The first learning loop.

```text
Predict
↓

Measure Error
↓

Backpropagation
↓

Update
↓

Repeat
```

Training reached:

```text
Train Loss ≈ 2.45
Validation Loss ≈ 2.61
```

Output looked like:

```text
ONBRDrk;
Ange akind!

...
```

Not coherent yet — expected for Bigram.

But:
- Learned punctuation ✅
- Learned word structure ✅
- Learned Shakespeare formatting ✅

---

## 5. Attention Intuition (Commit 6)

Before self-attention:

All previous tokens contribute equally.

```text
Past Tokens

↓

Average

↓

Prediction
```

Three implementations:

### Version 1
Naive loops

### Version 2
Matrix multiplication

### Version 3
Softmax weighting

Key insight:

```text
Uniform averaging

↓

Learned attention weights

↓

Self-Attention
```

The secret:

```text
Attention = softmax(QKᵀ)V
```

---

# Project Structure

```text
loom-gpt/

data/
└── input.txt

src/
├── tokenizer.py
├── dataset.py
├── bigram.py

notebooks/
└── averaging_trick.py

train_bigram.py

README.md
```

---

# Concepts Mastered So Far

✔ Tokenization  
✔ Embeddings  
✔ Tensors  
✔ Dataset batching  
✔ Bigram Language Models  
✔ Cross Entropy Loss  
✔ Backpropagation  
✔ Training loops  
✔ Matrix multiplication  
✔ Softmax  
✔ Attention intuition  

---

# Current Status

```text
Language Understanding:
███░░░░░░░░ 30%

Model Capability:
██░░░░░░░░░ 20%

Transformer Progress:
████░░░░░░ 40%
```

---

# Next Milestone

→ Build actual causal self-attention

Goal:

```text
Current:
One Token
↓

Next Token

Next:

All Previous Tokens
↓

Next Token
```

---

# References

Andrej Karpathy  
Let's Build GPT: From Scratch

Tiny Shakespeare Dataset

---

