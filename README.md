# LOOM-GPT

Train small GPT-style transformers on your own text, notes, code, and structured files.

LOOM-GPT started as a from-scratch implementation inspired by Andrej Karpathy's
["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial. It is now
growing into a local AI laboratory: a readable toolkit for preparing datasets,
training tiny domain-specific models, and experimenting with how models can be
combined.

The project is intentionally small. It is useful for learning, prototyping, and
running controlled experiments. It is not a replacement for a large pretrained
assistant.

## Current Release: Dataset Loom

The first usable milestone lets you:

- Prepare a dataset from a file or folder.
- Ingest `.txt`, `.md`, `.jsonl`, `.csv`, and common source-code files.
- Inspect a generated dataset manifest.
- Train with `tiny`, `laptop`, or `single_gpu` presets.
- Choose universal UTF-8 byte tokenization or the original character tokenizer.
- Save self-describing checkpoints that remember their model configuration.
- Stop early when validation quality no longer improves.
- Export `history.csv` metrics for charts and experiment reports.
- Resume interrupted training runs.
- Use reproducible random seeds and generation presets.
- Generate text from a trained checkpoint.
- Blend compatible specialist checkpoints with `loom weave`.

## Quick Start

Create a virtual environment and install the project:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Prepare a dataset from your own folder:

```bash
loom dataset add ./my-notes --name notes
loom dataset inspect notes
```

Train a small model:

```bash
loom train --data data/loom/notes/input.txt --out out/notes --preset tiny
```

For a longer run with early stopping:

```bash
loom train \
  --data data/loom/notes/input.txt \
  --out out/notes \
  --preset laptop \
  --early-stopping 8 \
  --seed 42
```

Resume an interrupted run:

```bash
loom train \
  --data data/loom/notes/input.txt \
  --out out/notes \
  --preset laptop \
  --max-iters 5000 \
  --resume out/notes/final_model.pt
```

Generate text:

```bash
loom generate --checkpoint out/notes/best_model.pt --prompt "Today I learned" --preset precise
```

Blend two or more specialist checkpoints:

```bash
loom weave \
  --model poetry=out/poetry/best_model.pt \
  --model technology=out/technology/best_model.pt \
  --weight poetry=0.7 \
  --weight technology=0.3 \
  --prompt "The city at night" \
  --trace-out out/weaving/city-trace.json
```

You can also run commands without installing the CLI:

```bash
python loom.py dataset add ./my-notes --name notes
python loom.py train --data data/loom/notes/input.txt --out out/notes --preset tiny
```

## Dataset Preparation

LOOM combines supported files into one training corpus and records a manifest:

```text
data/loom/notes/
  input.txt
  manifest.json
```

Each source file receives a boundary marker:

```text
<loom:file path="docs/example.md">
file contents
</loom:file>
```

The markers retain useful file context for experiments with notes and code.
Prepared corpora are ignored by Git because personal datasets may be large or
sensitive.

## Tokenizers

Byte tokenization is the default:

```bash
loom train --data data/loom/notes/input.txt --tokenizer byte
```

It uses a fixed vocabulary of 256 UTF-8 bytes, so the same model pipeline works
with multilingual text, code, and mixed datasets. The original educational
character tokenizer remains available:

```bash
loom train --data data/input.txt --tokenizer char
```

## Presets

| Preset | Intended use | Layers | Heads | Embedding size |
| --- | --- | ---: | ---: | ---: |
| `tiny` | Quick experiments | 2 | 2 | 64 |
| `laptop` | Default local training | 4 | 4 | 128 |
| `single_gpu` | Longer GPU runs | 6 | 6 | 384 |

Use `--max-iters` to override the preset training duration:

```bash
loom train --data data/loom/notes/input.txt --preset tiny --max-iters 50
```

Training writes `best_model.pt`, `final_model.pt`, and `history.csv` under the
selected output folder. `best_model.pt` is usually the right checkpoint for
generation because it preserves the lowest validation loss before overfitting.

## Generation Presets

| Preset | Temperature | Top-k | Use case |
| --- | ---: | ---: | --- |
| `precise` | 0.5 | 15 | More conservative output |
| `balanced` | 0.8 | 40 | Default experiments |
| `creative` | 1.0 | 80 | More varied output |

Override either value manually when needed:

```bash
loom generate \
  --checkpoint out/notes/best_model.pt \
  --prompt "Artificial intelligence can " \
  --temperature 0.6 \
  --top-k 20
```

## Architecture

The model remains a readable decoder-only transformer built from scratch:

```text
input files
  -> dataset preparation
  -> tokenizer
  -> token and position embeddings
  -> causal multi-head self-attention
  -> feed-forward layers
  -> next-token prediction
  -> generated text
```

Important files:

```text
loom.py              CLI entry point
config.py            Training presets
src/data_prep.py     Dataset ingestion and manifests
src/tokenizer.py     Byte and character tokenizers
src/attention.py     Causal self-attention
src/model.py         Transformer blocks and GPT model
train.py             Training and checkpoints
generate.py          Text generation
tests/               Lightweight tests
```

## Model Weaving

The signature feature is **Model Weaving**: train small specialists on different
datasets and blend their influence during generation.

```text
poetry expert      70% --\
philosophy expert  30% ----> woven generation
code expert         0% --/
```

Planned milestones:

- Add a web dashboard with dataset stats, loss charts, and generation controls.
- Train separate domain specialists. Current CLI support is available through `loom weave`.
- Blend specialists with manual weights.
- Visualize each specialist's influence token by token.
- Compare manual blending against a learned router.
- Evaluate domain interference, model size, memory use, and generation quality.

Current weaving constraints:

- Specialists must use the default byte tokenizer.
- Specialists must have the same architecture.
- Legacy character-tokenizer checkpoints are not supported for weaving.
- The optional JSON trace records which specialist most influenced each generated token.

## Development Workflow

Build and push the project in reviewable milestones:

```bash
# Step 1: reusable dataset framework and byte tokenizer
git add loom.py pyproject.toml config.py train.py generate.py src/data_prep.py src/tokenizer.py tests .gitignore README.md
git commit -m "feat: add reusable dataset training framework"
git push origin main

# Step 2: dashboard
git add loom/dashboard.py requirements.txt README.md
git commit -m "feat: add local training dashboard"
git push origin main

# Step 3: model weaving
git add src/weaving.py loom/dashboard.py tests README.md
git commit -m "feat: add weighted model weaving"
git push origin main

# Step 4: research evaluation
git add experiments docs README.md
git commit -m "feat: add model weaving evaluation suite"
git push origin main
```

Before each push:

```bash
python -m unittest discover -s tests -v
git status
git diff --stat
```

Review the staged files before committing. Do not commit private datasets or
generated model checkpoints.

## Legacy Checkpoints

Older checkpoints only contain model weights. When generating from one, provide
the old dataset and character tokenizer explicitly:

```bash
loom generate \
  --checkpoint out/best_model.pt \
  --data data/input.txt \
  --tokenizer char \
  --prompt "ROMEO:"
```

New checkpoints store their tokenizer and architecture configuration
automatically.
