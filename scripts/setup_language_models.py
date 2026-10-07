"""Download only the pinned public translation runtime; never download datasets."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'a7d96a16729f812578bd55e7366147beda625d86'
FILES = ('README.md', 'config.json', 'generation_config.json', 'pytorch_model.bin',
         'source.spm', 'target.spm', 'tokenizer_config.json', 'vocab.json')


def main():
    directory = ROOT / 'models/language_tools/opus-mt-hi-en'
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'configs/language_tools_manifest.json').read_text())
    for name in FILES:
        path = directory / name
        expected = manifest['translate_hi_en']['files'][name]
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        url = f'https://huggingface.co/Helsinki-NLP/opus-mt-hi-en/resolve/{REVISION}/{name}'
        temporary = path.with_suffix(path.suffix + '.tmp')
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            temporary.unlink()
            raise ValueError(f'Download checksum mismatch: {name}')
        temporary.replace(path)
        print(f'Verified {name}')


if __name__ == '__main__':
    main()
