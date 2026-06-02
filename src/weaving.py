"""Weighted Model Weaving for compatible LOOM checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ARCHITECTURE_KEYS = ('tokenizer', 'vocab_size', 'block_size', 'n_layer', 'n_head', 'n_embd')


@dataclass(frozen=True)
class SpecialistSpec:
    name: str
    checkpoint: str


def parse_assignment(raw: str, label: str) -> tuple[str, str]:
    if '=' not in raw:
        raise ValueError(f"{label} must use name=value format: {raw}")
    name, value = raw.split('=', 1)
    name = name.strip()
    value = value.strip()
    if not name or not value:
        raise ValueError(f"{label} must include both name and value: {raw}")
    return name, value


def normalize_weights(names: list[str], raw_weights: dict[str, float] | None = None) -> list[float]:
    if len(set(names)) != len(names):
        raise ValueError('Specialist names must be unique.')
    raw_weights = raw_weights or {}
    unknown = sorted(set(raw_weights) - set(names))
    if unknown:
        raise ValueError(f"Weight provided for unknown specialist(s): {', '.join(unknown)}")
    if not raw_weights:
        return [1.0 / len(names)] * len(names)
    weights = [float(raw_weights.get(name, 0.0)) for name in names]
    if any(weight < 0 for weight in weights):
        raise ValueError('Weights must be non-negative.')
    total = sum(weights)
    if total <= 0:
        raise ValueError('At least one specialist weight must be positive.')
    return [weight / total for weight in weights]


def validate_compatible_configs(configs: list[dict]) -> None:
    if len(configs) < 2:
        raise ValueError('Model weaving requires at least two specialists.')
    reference = configs[0]
    if reference.get('tokenizer') != 'byte':
        raise ValueError("Model weaving currently supports byte-tokenizer checkpoints only.")
    for config in configs[1:]:
        for key in ARCHITECTURE_KEYS:
            if config.get(key) != reference.get(key):
                raise ValueError(
                    f"Incompatible specialists: '{key}' differs "
                    f"({reference.get(key)!r} != {config.get(key)!r})."
                )


def load_trace(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _import_torch():
    import torch
    import torch.nn.functional as F

    return torch, F


def load_specialists(specs: list[SpecialistSpec], device: str | None = None):
    torch, _ = _import_torch()
    from src.model import GPT

    loaded = []
    configs = []
    target_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    for spec in specs:
        checkpoint = torch.load(spec.checkpoint, map_location=target_device)
        if 'model_state' not in checkpoint or 'config' not in checkpoint:
            raise ValueError(f"{spec.name} is a legacy checkpoint and cannot be woven.")
        config = checkpoint['config']
        configs.append(config)
        model = GPT(
            vocab_size=config['vocab_size'],
            n_embd=config['n_embd'],
            n_heads=config['n_head'],
            n_layer=config['n_layer'],
            block_size=config['block_size'],
            dropout=0.0,
        )
        model.load_state_dict(checkpoint['model_state'])
        model.to(target_device)
        model.eval()
        loaded.append((spec.name, model))
    validate_compatible_configs(configs)
    return loaded, configs[0]


def blend_logits(logits_by_model, weights):
    return sum(weight * logits for weight, logits in zip(weights, logits_by_model))


def contribution_for_token(probabilities_by_model, weights, token_id: int) -> list[float]:
    raw = [
        float(probs[0, token_id].item()) * weight
        for probs, weight in zip(probabilities_by_model, weights)
    ]
    total = sum(raw)
    if total <= 0:
        return [0.0 for _ in raw]
    return [value / total for value in raw]


def generate_woven(
    specialists,
    context,
    weights: list[float],
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 40,
):
    torch, F = _import_torch()
    trace = []
    names = [name for name, _ in specialists]

    for _ in range(max_new_tokens):
        logits_by_model = []
        probabilities_by_model = []
        for _, model in specialists:
            idx_cond = context[:, -model.block_size:]
            logits, _ = model(idx_cond)
            next_logits = logits[:, -1, :]
            logits_by_model.append(next_logits)
            probabilities_by_model.append(F.softmax(next_logits, dim=-1))

        woven_logits = blend_logits(logits_by_model, weights) / temperature
        if top_k is not None:
            values, _ = torch.topk(woven_logits, min(top_k, woven_logits.size(-1)))
            woven_logits[woven_logits < values[:, [-1]]] = float('-inf')

        probs = F.softmax(woven_logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        token_id = int(idx_next.item())
        contributions = contribution_for_token(probabilities_by_model, weights, token_id)
        strongest_index = max(range(len(contributions)), key=contributions.__getitem__)
        trace.append({
            'token_id': token_id,
            'specialist': names[strongest_index],
            'contributions': dict(zip(names, contributions)),
        })
        context = torch.cat([context, idx_next], dim=1)

    return context, trace
