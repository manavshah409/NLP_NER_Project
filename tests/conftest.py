"""Tests use synthetic fixtures; actual project datasets remain sealed."""
import builtins
import os
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def seal_real_datasets(monkeypatch):
    original_builtin, original_path, original_os = builtins.open, Path.open, os.open

    def check(path):
        if isinstance(path, int):
            return
        resolved = Path(os.fsdecode(path)).resolve()
        if resolved.is_relative_to(ROOT / 'data'):
            raise PermissionError(f'Test suite cannot access real datasets: {resolved.name}')
        if resolved.is_relative_to(ROOT / 'reports') and (
                resolved.suffix == '.jsonl' or 'restricted_examples' in resolved.parts
                or resolved.name in ('error_analysis.json', 'error_analysis.md')):
            raise PermissionError('Test suite cannot access record-level benchmark reports')

    def builtin_open(path, *args, **kwargs):
        check(path)
        return original_builtin(path, *args, **kwargs)

    def path_open(path, *args, **kwargs):
        check(path)
        return original_path(path, *args, **kwargs)

    def os_open(path, *args, **kwargs):
        check(path)
        return original_os(path, *args, **kwargs)

    monkeypatch.setattr(builtins, 'open', builtin_open)
    monkeypatch.setattr(Path, 'open', path_open)
    monkeypatch.setattr(os, 'open', os_open)
