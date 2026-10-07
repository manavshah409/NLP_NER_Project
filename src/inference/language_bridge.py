"""Bounded, local-only IPC to the optional isolated language environment."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def run_language_task(task, text):
    if task not in ('english_ner', 'translate_hi_en'):
        raise ValueError('Unsupported language task.')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Enter non-empty text.')
    if len(text) > 5000:
        raise ValueError('Maximum input length is 5,000 Unicode code points.')
    python = ROOT / '.venv-language/bin/python'
    if not python.is_file():
        raise RuntimeError('Language environment missing. Follow docs/demo/LANGUAGE_TOOLS.md to install the optional local models.')
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
    try:
        result = subprocess.run([str(python), '-m', 'src.language_tools.worker'],
                                input=json.dumps({'task': task, 'text': text}),
                                text=True, capture_output=True, cwd=ROOT, env=env, timeout=120)
    except subprocess.TimeoutExpired:
        raise RuntimeError('Local language processing exceeded 120 seconds. Try a shorter input.') from None
    if result.returncode:
        # Never return captured stderr: dependencies can echo user input there.
        raise RuntimeError('Language model could not run. Check the isolated installation and model checksums using docs/demo/LANGUAGE_TOOLS.md.')
    payload = json.loads(result.stdout)
    if 'error' in payload:
        raise ValueError(payload['error'])
    return payload
