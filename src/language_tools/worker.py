"""One-shot offline worker. Input/output use pipes, never files or remote APIs."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / 'models/language_tools/opus-mt-hi-en'
LABEL_MAP = {'PERSON': 'PER', 'ORG': 'ORG', 'GPE': 'LOC', 'LOC': 'LOC'}


def verify_artifacts(task):
    manifest = json.loads((ROOT / 'configs/language_tools_manifest.json').read_text())
    group = manifest[task]
    for package, expected in manifest['packages'].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError('Language dependency version mismatch. Restore the isolated environment.')
    base = MODEL_DIR if task == 'translate_hi_en' else Path(importlib.metadata.distribution('en-core-web-sm').locate_file('en_core_web_sm'))
    for name, expected in group['files'].items():
        path = base / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Language model checksum failed. Restore the pinned original model files.')
    return group


def english_ner(text):
    metadata = verify_artifacts('english_ner')
    import spacy
    nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
    doc = nlp(text)
    tokens = [{'text': t.text, 'start_char': t.idx, 'end_char': t.idx + len(t), 'index': t.i} for t in doc]
    labels = ['O'] * len(doc)
    entities = []
    for span in doc.ents:
        kind = LABEL_MAP.get(span.label_)
        if kind is None:
            continue
        bio = [f'B-{kind}'] + [f'I-{kind}'] * (len(span)-1)
        labels[span.start:span.end] = bio
        entity = dict(text=span.text, type=kind, source_label=span.label_, start_char=span.start_char,
                      end_char=span.end_char, start_token=span.start, end_token=span.end,
                      bio_sequence=bio, confidence=None)
        assert text[entity['start_char']:entity['end_char']] == entity['text']
        entities.append(entity)
    return dict(text=text, language='en', model_family='spaCy English NER', experiment_id=metadata['model_id'],
                offset_convention='zero-based, end-exclusive Unicode code points', tokens=tokens,
                predicted_labels=labels, entities=entities,
                diagnostics={'token_count': len(tokens), 'device': 'cpu', 'checksum_status': 'verified'},
                limitations='Pretrained external model; no project validation score. PERSON→PER, ORG→ORG, GPE/LOC→LOC; other labels excluded. Not a controlled comparison to the Hindi models.')


def translate_hi_en(text):
    metadata = verify_artifacts('translate_hi_en')
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    tokenizer = MarianTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = MarianMTModel.from_pretrained(MODEL_DIR, local_files_only=True).eval()
    model.requires_grad_(False)
    # Refuse oversized input; never silently truncate or manufacture alignment.
    inputs = tokenizer(text, return_tensors='pt', truncation=False)
    if inputs['input_ids'].shape[1] > 512:
        raise ValueError('Translation is limited to 512 model tokens. Split the text into shorter passages.')
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, num_beams=4, max_new_tokens=512)
    if output[0, -1].item() != tokenizer.eos_token_id:
        raise ValueError('Translation reached its output limit. Use a shorter passage; incomplete output was discarded.')
    translated = tokenizer.decode(output[0], skip_special_tokens=True)
    return dict(text=text, translated_text=translated, source_language='hi', target_language='en',
                model_id=metadata['model_id'], revision=metadata['revision'], device='cpu',
                checksum_status='verified', alignment=None,
                limitations='Machine translation may alter names, facts and meaning. No project translation-quality evaluation. Original Hindi NER offsets do not apply to translated text.')


def main():
    # Install before model or dataset reads. No project datasets are needed.
    from src.development_guard import install
    install()
    request = json.loads(sys.stdin.read(100000))
    text = request.get('text')
    try:
        if not isinstance(text, str) or not text.strip() or len(text) > 5000:
            raise ValueError('Enter 1–5,000 Unicode code points of non-empty text.')
        task = request.get('task')
        started = time.perf_counter()
        if task == 'english_ner':
            result = english_ner(text)
        elif task == 'translate_hi_en':
            result = translate_hi_en(text)
        else:
            raise ValueError('Unsupported language task.')
        result['elapsed_ms_including_model_load'] = round((time.perf_counter()-started)*1000, 2)
    except ValueError as error:
        result = {'error': str(error)}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
