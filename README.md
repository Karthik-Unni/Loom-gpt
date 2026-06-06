#  🧵 LOOM-GPT

Train small specialist transformers locally. Weave their outputs together. Inspect which specialist shaped each generated token.

LOOM-GPT is a local transformer laboratory for students, developers, writers, and researchers who want to understand and experiment with GPT-style models from the inside.

It started as a from-scratch PyTorch implementation inspired by Andrej Karpathy's "Let's build GPT" tutorial. It is now becoming **LOOM Studio**: a framework where users can prepare their own datasets, train compact specialist models, and blend those specialists during generation.

LOOM-GPT is not a ChatGPT replacement. It does not use a giant pretrained model. Instead, it gives you a readable, hackable, local system for training tiny domain-specific transformers and studying how they behave.

## What This Project Does

LOOM-GPT lets you:

- Prepare a dataset from your own files and folders.
- Train a small GPT-style transformer from scratch.
- Save reusable checkpoints with model configuration included.
- Track training and validation loss in `history.csv`.
- Stop training early when validation loss stops improving.
- Generate text from one trained specialist.
- Train multiple specialists on different datasets.
- Weave specialists by blending their next-token predictions.
- Export a JSON trace showing which specialist most influenced each generated token.

The core workflow looks like this:

```text
Your files
  -> dataset preparation
  -> byte tokenization
  -> specialist training
  -> checkpoint
  -> generation
  -> optional Model Weaving
```

## Who Can Use It?

LOOM-GPT is useful for:

- **Students** learning how GPT models work without hiding everything behind an API.
- **Developers** experimenting with small domain-specific text models.
- **Writers** training tiny style models on different genres or voices.
- **Researchers** testing interpretable model composition ideas.
- **Educators** demonstrating tokenization, attention, overfitting, validation loss, and sampling.

Example user stories:

- A student trains one specialist on poetry and another on technical documentation, then blends the two to see how generation changes.
- A developer trains a tiny model on internal notes or code comments to study local domain language.
- A researcher compares one mixed-data model against several woven specialist models.
- A teacher uses the training logs to show why validation loss matters more than training loss.

## Key Features

### Custom Dataset Preparation

Point LOOM at a file or folder:

```bash
loom dataset add ./my-notes --name notes
```

LOOM combines supported files into:

```text
data/loom/notes/
  input.txt
  manifest.json
```

Supported file types include:

- `.txt`
- `.md`
- `.jsonl`
- `.csv`
- Common code files such as `.py`, `.js`, `.ts`, `.java`, `.rs`, `.go`, `.html`, `.css`, `.sql`, `.yaml`

Each source file is wrapped with a boundary marker:

```text
<loom:file path="docs/example.md">
file contents
</loom:file>
```

That keeps file context visible to the model and to future experiments.

### Local Transformer Training

Train a small decoder-only GPT model:

```bash
loom train --data data/loom/notes/input.txt --out out/notes --preset tiny
```

Longer training with early stopping:

```bash
loom train \
  --data data/loom/notes/input.txt \
  --out out/notes \
  --preset laptop \
  --max-iters 5000 \
  --early-stopping 8 \
  --seed 42
```

Training creates:

```text
out/notes/
  best_model.pt
  final_model.pt
  history.csv
```

Use `best_model.pt` for generation because it stores the checkpoint with the lowest validation loss.

### Training Presets

| Preset | Use case | Layers | Heads | Embedding size |
| --- | --- | ---: | ---: | ---: |
| `tiny` | Quick smoke tests | 2 | 2 | 64 |
| `laptop` | Normal local experiments | 4 | 4 | 128 |
| `single_gpu` | Longer GPU runs | 6 | 6 | 384 |

### Byte Tokenization

LOOM uses UTF-8 byte tokenization by default:

```text
text -> bytes -> token IDs from 0 to 255
```

This means the same training pipeline can handle English, multilingual text, code, and mixed folders.

The original character tokenizer is still available for educational experiments:

```bash
loom train --data data/input.txt --tokenizer char
```

### Generation

Generate from a single trained specialist:

```bash
loom generate \
  --checkpoint out/notes/best_model.pt \
  --prompt "Today I learned that " \
  --preset precise \
  --tokens 250
```

Generation presets:

| Preset | Temperature | Top-k | Behavior |
| --- | ---: | ---: | --- |
| `precise` | 0.5 | 15 | More conservative |
| `balanced` | 0.8 | 40 | Default |
| `creative` | 1.0 | 80 | More varied |

Manual override:

```bash
loom generate \
  --checkpoint out/notes/best_model.pt \
  --prompt "Artificial intelligence can " \
  --temperature 0.6 \
  --top-k 20
```

## Model Weaving

Model Weaving is LOOM-GPT's signature feature.

Instead of training one model on everything, you train separate specialists:

```text
poetry specialist
technology specialist
philosophy specialist
```

During generation, LOOM asks each specialist for its next-token prediction, blends their logits using your weights, samples one token, and repeats.

```text
Prompt
  -> poetry logits
  -> technology logits
  -> philosophy logits
  -> weighted blend
  -> sampled token
  -> influence trace
```

Simple example:

```text
poetry      70%
technology 30%

Prompt: "The city at night"
```

LOOM blends the specialists like this:

```python
woven_logits = 0.7 * poetry_logits + 0.3 * technology_logits
```

The result is not just one model generating text. It is several small models contributing to the next token.

## Example Output

Poetry specialist only:
> "The city at night breathes slow, each lamp a word
> the darkness reads and folds into its keeping."

Technology specialist only:
> "The city at night runs distributed processes across
> nodes, latency rising as packet loss compounds at scale."

Woven 70% poetry / 30% technology:
> "The city at night holds its network close, each signal
> a quiet pulse the infrastructure learns to trust."

### Weaving Command

```bash
loom weave \
  --model poetry=out/poetry/best_model.pt \
  --model technology=out/technology/best_model.pt \
  --weight poetry=0.7 \
  --weight technology=0.3 \
  --prompt "The city at night" \
  --tokens 300 \
  --preset balanced \
  --trace-out out/weaving/city-trace.json
```

If no weights are provided, LOOM gives all specialists equal weight.

```bash
loom weave \
  --model poetry=out/poetry/best_model.pt \
  --model technology=out/technology/best_model.pt \
  --prompt "The city at night"
```

### Influence Trace

When you pass `--trace-out`, LOOM writes a JSON file like:

```json
[
  {
    "token_id": 84,
    "specialist": "poetry",
    "contributions": {
      "poetry": 0.72,
      "technology": 0.28
    }
  }
]
```

Each item tells you:

- The generated token ID.
- Which specialist had the strongest contribution.
- Each specialist's normalized contribution for that token.

This trace is the foundation for the future dashboard visualization where generated tokens can be colored by specialist influence.

### Current Weaving Constraints

For now:

- Specialists must use the default `byte` tokenizer.
- Specialists must have the same architecture.
- Legacy character-tokenizer checkpoints cannot be woven.
- Weaving works best when specialists were trained with the same preset.

Recommended specialist training:

```bash
loom train --data data/loom/poetry/input.txt --out out/poetry --preset laptop --early-stopping 8
loom train --data data/loom/technology/input.txt --out out/technology --preset laptop --early-stopping 8
loom train --data data/loom/philosophy/input.txt --out out/philosophy --preset laptop --early-stopping 8
```

Then weave:

```bash
loom weave \
  --model poetry=out/poetry/best_model.pt \
  --model technology=out/technology/best_model.pt \
  --model philosophy=out/philosophy/best_model.pt \
  --weight poetry=0.5 \
  --weight technology=0.3 \
  --weight philosophy=0.2 \
  --prompt "The future belongs to "
```

## Complete Example Use Case

Imagine a student wants to explore how style changes when technical writing and poetry are blended.

Create two folders:

```text
demo-data/
  poetry/
    poems.txt
  technology/
    ai-notes.md
    software-docs.txt
```

Prepare datasets:

```bash
loom dataset add ./demo-data/poetry --name poetry
loom dataset add ./demo-data/technology --name technology
```

Train specialists:

```bash
loom train --data data/loom/poetry/input.txt --out out/poetry --preset laptop --early-stopping 8
loom train --data data/loom/technology/input.txt --out out/technology --preset laptop --early-stopping 8
```

Generate from each specialist separately:

```bash
loom generate --checkpoint out/poetry/best_model.pt --prompt "The city at night" --preset precise
loom generate --checkpoint out/technology/best_model.pt --prompt "The city at night" --preset precise
```

Now weave them:

```bash
loom weave \
  --model poetry=out/poetry/best_model.pt \
  --model technology=out/technology/best_model.pt \
  --weight poetry=0.8 \
  --weight technology=0.2 \
  --prompt "The city at night" \
  --trace-out out/weaving/poetic-city.json
```

Then flip the weights:

```bash
loom weave \
  --model poetry=out/poetry/best_model.pt \
  --model technology=out/technology/best_model.pt \
  --weight poetry=0.2 \
  --weight technology=0.8 \
  --prompt "The city at night" \
  --trace-out out/weaving/technical-city.json
```

The user can compare:

- Poetry-only output
- Technology-only output
- Mostly-poetry woven output
- Mostly-technology woven output
- Token influence traces

That is the main product idea: train local specialists, control their blend, and inspect how the blend shapes generation.

## Installation

```bash
git clone https://github.com/Karthik-Unni/Loom-gpt.git
cd Loom-gpt
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
```

## Commands

Prepare a dataset:

```bash
loom dataset add ./my-notes --name notes
loom dataset inspect notes
```

Train:

```bash
loom train --data data/loom/notes/input.txt --out out/notes --preset laptop
```

Resume:

```bash
loom train \
  --data data/loom/notes/input.txt \
  --out out/notes \
  --preset laptop \
  --resume out/notes/final_model.pt
```

Generate:

```bash
loom generate --checkpoint out/notes/best_model.pt --prompt "Today I learned"
```

Weave:

```bash
loom weave \
  --model a=out/a/best_model.pt \
  --model b=out/b/best_model.pt \
  --weight a=0.6 \
  --weight b=0.4 \
  --prompt "Once upon a system"
```

## Architecture

The model is a small decoder-only transformer built from scratch in PyTorch:

```text
tokens
  -> token embeddings
  -> position embeddings
  -> causal multi-head self-attention
  -> feed-forward layers
  -> layer normalization
  -> next-token logits
```

Important files:

```text
loom.py              Main CLI wrapper
train.py             Training entry point
generate.py          Single-checkpoint generation
weave.py             Multi-specialist weaving entry point
config.py            Model presets
src/model.py         GPT model
src/attention.py     Causal self-attention
src/tokenizer.py     Byte and character tokenizers
src/data_prep.py     Dataset ingestion
src/training.py      Early stopping, history, generation presets
src/weaving.py       Weighted Model Weaving
tests/               Unit tests
```

## What LOOM-GPT Is Good At

- Learning transformer internals.
- Running small local experiments.
- Comparing datasets and specialists.
- Demonstrating overfitting and validation loss.
- Exploring controllable generation through weighted specialists.
- Creating a portfolio project with a clear research-style idea.

## What LOOM-GPT Is Not

- It is not ChatGPT.
- It is not a factual assistant.
- It is not trained on internet-scale data.
- It will not produce polished text from tiny datasets.
- It does not yet have a full dashboard.

Small models trained from scratch need clean data and patience. The goal is experimentation and interpretability, not production-grade language understanding.

## Recommended Data Size

For experiments:

```text
100,000+ characters: basic behavior
500,000+ characters: better small-model experiments
2,000,000+ characters: noticeably stronger local style learning
```

Use clean, consistent data. Remove broken HTML, duplicated lines, unrelated text, and noisy formatting when possible.

```text
Datasets -> Train -> Generate -> Weave -> Metrics
```

The long-term vision is a local LOOM Studio interface where users train specialists, move sliders, generate text, and see which specialist influenced each token.


This is still an expirimental grade project!

