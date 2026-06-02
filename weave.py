import argparse
import json
from pathlib import Path

import torch

from src.tokenizer import create_tokenizer
from src.training import resolve_generation_settings
from src.weaving import (
    SpecialistSpec,
    generate_woven,
    load_specialists,
    normalize_weights,
    parse_assignment,
)


parser = argparse.ArgumentParser(description='Blend compatible LOOM specialist checkpoints.')
parser.add_argument('--model', action='append', required=True, help='Specialist checkpoint as name=path. Repeat this flag.')
parser.add_argument('--weight', action='append', default=[], help='Specialist weight as name=value. Repeat this flag.')
parser.add_argument('--prompt', default='')
parser.add_argument('--tokens', type=int, default=300)
parser.add_argument('--preset', choices=['precise', 'balanced', 'creative'], default='balanced')
parser.add_argument('--temperature', type=float)
parser.add_argument('--top-k', dest='top_k', type=int)
parser.add_argument('--trace-out', help='Optional JSON path for token influence trace.')
args = parser.parse_args()

temperature, top_k = resolve_generation_settings(args.preset, args.temperature, args.top_k)
model_pairs = [parse_assignment(raw, '--model') for raw in args.model]
weight_pairs = dict(
    (name, float(value))
    for name, value in (parse_assignment(raw, '--weight') for raw in args.weight)
)
specs = [SpecialistSpec(name, checkpoint) for name, checkpoint in model_pairs]
weights = normalize_weights([spec.name for spec in specs], weight_pairs)

specialists, config = load_specialists(specs)
tokenizer = create_tokenizer(config['tokenizer'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.prompt:
    context = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long)[None].to(device)
else:
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

print('--- loom-gpt weaving ---')
for spec, weight in zip(specs, weights):
    print(f'{spec.name:16s} {weight:.2%}  {spec.checkpoint}')
print(f'temp={temperature} | top_k={top_k}\n')

if args.prompt:
    print(args.prompt, end='', flush=True)
output, trace = generate_woven(specialists, context, weights, args.tokens, temperature, top_k)
generated_text = tokenizer.decode(output[0].tolist())
if args.prompt:
    print(generated_text[len(args.prompt):])
else:
    print(generated_text)

if args.trace_out:
    trace_path = Path(args.trace_out)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace, indent=2) + '\n', encoding='utf-8')
    print(f'\nTrace: {trace_path}')
