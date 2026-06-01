"""Dataset ingestion utilities for LOOM-GPT."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


SUPPORTED_EXTENSIONS = {
    '.txt', '.md', '.rst', '.jsonl', '.csv',
    '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.c', '.cpp', '.h',
    '.go', '.rs', '.html', '.css', '.sql', '.yaml', '.yml',
}


@dataclass
class DatasetManifest:
    name: str
    source: str
    output_file: str
    created_at: str
    file_count: int
    character_count: int
    byte_count: int
    sha256: str
    extensions: dict[str, int]


def _string_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _string_values(child)


def _read_file(path: Path) -> str:
    if path.suffix.lower() == '.jsonl':
        chunks = []
        for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                chunks.extend(_string_values(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSONL in {path} at line {line_number}') from exc
        return '\n'.join(chunks)

    if path.suffix.lower() == '.csv':
        with path.open('r', encoding='utf-8', newline='') as handle:
            return '\n'.join(
                value
                for row in csv.reader(handle)
                for value in row
                if value
            )

    return path.read_text(encoding='utf-8')


def discover_files(source: str | Path) -> list[Path]:
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f'Dataset source does not exist: {source}')
    files = [source] if source.is_file() else list(source.rglob('*'))
    return sorted(
        path for path in files
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def prepare_dataset(source: str | Path, output_dir: str | Path, name: str) -> DatasetManifest:
    source = Path(source).resolve()
    output_dir = Path(output_dir)
    files = discover_files(source)
    if not files:
        supported = ', '.join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f'No supported files found. Supported extensions: {supported}')

    sections = []
    extensions: dict[str, int] = {}
    for path in files:
        extension = path.suffix.lower()
        extensions[extension] = extensions.get(extension, 0) + 1
        relative_path = Path(path.name) if source.is_file() else path.relative_to(source)
        sections.append(f'\n<loom:file path="{relative_path.as_posix()}">\n{_read_file(path)}\n</loom:file>\n')

    corpus = ''.join(sections)
    encoded = corpus.encode('utf-8')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'input.txt'
    output_file.write_text(corpus, encoding='utf-8')

    manifest = DatasetManifest(
        name=name,
        source=str(source),
        output_file=str(output_file.resolve()),
        created_at=datetime.now(timezone.utc).isoformat(),
        file_count=len(files),
        character_count=len(corpus),
        byte_count=len(encoded),
        sha256=hashlib.sha256(encoded).hexdigest(),
        extensions=dict(sorted(extensions.items())),
    )
    (output_dir / 'manifest.json').write_text(
        json.dumps(asdict(manifest), indent=2) + '\n',
        encoding='utf-8',
    )
    return manifest


def read_manifest(dataset_dir: str | Path) -> DatasetManifest:
    manifest_path = Path(dataset_dir) / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Manifest does not exist: {manifest_path}')
    return DatasetManifest(**json.loads(manifest_path.read_text(encoding='utf-8')))
