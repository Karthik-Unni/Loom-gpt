"""Small, testable helpers for LOOM training runs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


GENERATION_PRESETS = {
    'precise': {'temperature': 0.5, 'top_k': 15},
    'balanced': {'temperature': 0.8, 'top_k': 40},
    'creative': {'temperature': 1.0, 'top_k': 80},
}


@dataclass
class EarlyStopping:
    """Track validation improvements measured at evaluation intervals."""

    patience: int
    best_loss: float = float('inf')
    best_step: int = 0
    stale_evaluations: int = 0

    def update(self, step: int, validation_loss: float) -> bool:
        improved = validation_loss < self.best_loss
        if improved:
            self.best_loss = validation_loss
            self.best_step = step
            self.stale_evaluations = 0
        else:
            self.stale_evaluations += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.stale_evaluations >= self.patience


class HistoryLogger:
    """Append evaluation metrics to a CSV file for dashboards and reports."""

    def __init__(self, path: str | Path, append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not append or not self.path.exists():
            with self.path.open('w', newline='', encoding='utf-8') as handle:
                csv.writer(handle).writerow(['step', 'train_loss', 'val_loss'])

    def append(self, step: int, train_loss: float, val_loss: float) -> None:
        with self.path.open('a', newline='', encoding='utf-8') as handle:
            csv.writer(handle).writerow([step, train_loss, val_loss])


def resolve_generation_settings(
    preset: str = 'balanced',
    temperature: float | None = None,
    top_k: int | None = None,
) -> tuple[float, int]:
    if preset not in GENERATION_PRESETS:
        raise ValueError(f"Unknown generation preset '{preset}'.")
    settings = GENERATION_PRESETS[preset]
    return (
        settings['temperature'] if temperature is None else temperature,
        settings['top_k'] if top_k is None else top_k,
    )
