"""Synthetic fixtures only; sealed benchmark contents are never read."""
import subprocess
import sys
from pathlib import Path
import pytest


def probe(tmp_path, expression):
    clean = tmp_path / 'data/derived/naamapadam_hi_crf_v1'
    clean.mkdir(parents=True)
    for name in ('train_clean.jsonl', 'validation_clean.jsonl', 'test_clean.jsonl', 'official_test.jsonl'):
        (clean / name).write_text('{}\n')
    code = f'from src.development_guard import install; from pathlib import Path; import os; install({str(tmp_path)!r}); {expression}'
    return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)


@pytest.mark.parametrize('name', ['train_clean', 'validation_clean'])
def test_permitted_splits(tmp_path, name):
    p=tmp_path / f'data/derived/naamapadam_hi_crf_v1/{name}.jsonl'
    assert probe(tmp_path, f'Path({str(p)!r}).read_text()').returncode == 0


@pytest.mark.parametrize('relative', ['data/derived/naamapadam_hi_crf_v1/test_clean.jsonl', 'data/derived/naamapadam_hi_crf_v1/official_test.jsonl', 'data/raw/naamapadam_hi/hi_test.json', 'reports/crf/100k_final/test_clean_predictions.jsonl', 'reports/crf/100k_final/restricted_examples/example.json', 'reports/crf/100k_final/test_clean_metrics.json'])
@pytest.mark.parametrize('api', ['open({p}).read()', 'Path({p}).read_bytes()', 'os.open({p}, os.O_RDONLY)'])
def test_sealed_paths(tmp_path, relative, api):
    result=probe(tmp_path, api.format(p=repr(str(tmp_path/relative))))
    assert result.returncode != 0 and 'PermissionError' in result.stderr


def test_symlink_alias(tmp_path):
    alias=tmp_path/'alias.jsonl'; alias.symlink_to(tmp_path/'data/derived/naamapadam_hi_crf_v1/test_clean.jsonl')
    assert 'PermissionError' in probe(tmp_path, f'open({str(alias)!r})').stderr


def test_write_forbidden(tmp_path):
    p=tmp_path/'data/derived/naamapadam_hi_crf_v1/train_clean.jsonl'
    assert 'PermissionError' in probe(tmp_path, f'open({str(p)!r}, "w")').stderr


def test_child_process_forbidden(tmp_path):
    assert 'PermissionError' in probe(tmp_path, 'import subprocess; subprocess.run(["true"])').stderr


def test_historical_training_is_blocked_before_training():
    code = 'from src.development_guard import install; install(); from src.training.train_crf import train; train(1000)'
    result=subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode != 0 and 'Development data access is limited' in result.stderr


def test_historical_evaluation_rejects_test_before_loading():
    code = 'from src.development_guard import install; install(); from src.evaluation.evaluate_crf import evaluate; evaluate("missing.crfsuite", "test_clean.jsonl", {}, set(), "unused")'
    result=subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode != 0 and 'Only validation_clean' in result.stderr
