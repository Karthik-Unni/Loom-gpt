"""LOOM-GPT command line interface."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from src.data_prep import prepare_dataset, read_manifest


def _dataset_dir(name: str) -> Path:
    return Path('data') / 'loom' / name


def _run(script: str, forwarded: list[str]) -> None:
    command = [sys.executable, script, *forwarded]
    raise SystemExit(subprocess.call(command))


def add_dataset(args) -> None:
    output_dir = Path(args.output) if args.output else _dataset_dir(args.name)
    manifest = prepare_dataset(args.source, output_dir, args.name)
    print(f"Prepared dataset '{manifest.name}'")
    print(f'Files:      {manifest.file_count:,}')
    print(f'Characters: {manifest.character_count:,}')
    print(f'Bytes:      {manifest.byte_count:,}')
    print(f'Output:     {manifest.output_file}')


def inspect_dataset(args) -> None:
    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        dataset_dir = _dataset_dir(args.dataset)
    manifest = read_manifest(dataset_dir)
    print(json.dumps(asdict(manifest), indent=2))


def train(args) -> None:
    forwarded = ['--data-path', args.data, '--out-dir', args.out, '--preset', args.preset]
    if args.tokenizer:
        forwarded.extend(['--tokenizer', args.tokenizer])
    if args.max_iters is not None:
        forwarded.extend(['--max-iters', str(args.max_iters)])
    forwarded.extend(['--early-stopping', str(args.early_stopping), '--seed', str(args.seed)])
    if args.resume:
        forwarded.extend(['--resume', args.resume])
    _run('train.py', forwarded)


def generate(args) -> None:
    forwarded = [
        '--checkpoint', args.checkpoint,
        '--tokens', str(args.tokens),
        '--preset', args.preset,
        '--prompt', args.prompt,
    ]
    if args.temperature is not None:
        forwarded.extend(['--temperature', str(args.temperature)])
    if args.top_k is not None:
        forwarded.extend(['--top_k', str(args.top_k)])
    if args.data:
        forwarded.extend(['--data-path', args.data])
    if args.tokenizer:
        forwarded.extend(['--tokenizer', args.tokenizer])
    _run('generate.py', forwarded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='loom', description='Train tiny GPT models on your own text.')
    commands = parser.add_subparsers(dest='command', required=True)

    dataset = commands.add_parser('dataset', help='Prepare and inspect datasets.')
    dataset_commands = dataset.add_subparsers(dest='dataset_command', required=True)

    add = dataset_commands.add_parser('add', help='Combine supported files into a LOOM dataset.')
    add.add_argument('source', help='Input file or directory.')
    add.add_argument('--name', required=True, help='Dataset name.')
    add.add_argument('--output', help='Output directory. Defaults to data/loom/<name>.')
    add.set_defaults(func=add_dataset)

    inspect = dataset_commands.add_parser('inspect', help='Print a prepared dataset manifest.')
    inspect.add_argument('dataset', help='Dataset directory or name.')
    inspect.set_defaults(func=inspect_dataset)

    train_parser = commands.add_parser('train', help='Train a transformer.')
    train_parser.add_argument('--data', required=True, help='Prepared input.txt path.')
    train_parser.add_argument('--out', default='out', help='Checkpoint output directory.')
    train_parser.add_argument('--preset', choices=['tiny', 'laptop', 'single_gpu'], default='laptop')
    train_parser.add_argument('--tokenizer', choices=['byte', 'char'])
    train_parser.add_argument('--max-iters', type=int)
    train_parser.add_argument('--early-stopping', type=int, default=8, metavar='EVALS')
    train_parser.add_argument('--seed', type=int, default=42)
    train_parser.add_argument('--resume', help='Resume from a checkpoint.')
    train_parser.set_defaults(func=train)

    generate_parser = commands.add_parser('generate', help='Generate text from a checkpoint.')
    generate_parser.add_argument('--checkpoint', required=True)
    generate_parser.add_argument('--data', help='Override dataset path for legacy checkpoints.')
    generate_parser.add_argument('--tokenizer', choices=['byte', 'char'], help='Override tokenizer.')
    generate_parser.add_argument('--prompt', default='')
    generate_parser.add_argument('--tokens', type=int, default=500)
    generate_parser.add_argument('--preset', choices=['precise', 'balanced', 'creative'], default='balanced')
    generate_parser.add_argument('--temperature', type=float)
    generate_parser.add_argument('--top-k', dest='top_k', type=int)
    generate_parser.set_defaults(func=generate)

    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    arguments.func(arguments)


if __name__ == '__main__':
    main()
