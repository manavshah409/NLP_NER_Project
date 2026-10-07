"""Verify pinned runtime artifacts only; never open any dataset or evaluation rows."""
import importlib.metadata
from pathlib import Path
import yaml
from src.utils.io import digest, file_hash, read_json
from src.inference.crf_predictor import verify_runtime as verify_classical

ROOT = Path(__file__).resolve().parents[2]
CRF_ID = 'crf_100k_seed42_24dd05e84a26'
NEURAL_ID = 'bilstm_crf_100k_seed42_9af1d47db429'
CRF_MANIFEST = 'reports/crf/100k_baseline_freeze_manifest.json'
NEURAL_MANIFEST = 'reports/bilstm_crf/bilstm_crf_freeze_manifest.json'
CRF_DIGEST = 'd9be51a8460b68dd51265f806f3ed2cb361168cc0db5006b2b42fb4eb780df88'
NEURAL_DIGEST = '758ed03e196c85f31116c072c41256a97e3b0086c7983c28a892aabf860c5abc'
LABELS = ('O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC')
COMMON_FILES = ('src/__init__.py', 'src/data/__init__.py', 'src/models/__init__.py',
                'src/utils/__init__.py', 'src/inference/__init__.py', 'src/utils/io.py',
                'src/data/bio.py', 'src/inference/text_preprocessor.py',
                'src/inference/entity_formatter.py', 'src/inference/exporter.py')


def envelope(root, path, expected_digest, experiment):
    value = read_json(Path(root) / path)
    payload = value['payload']
    if value['payload_sha256'] != expected_digest or digest(payload) != expected_digest:
        raise ValueError(f'Freeze manifest checksum mismatch: {path}. Restore the original manifest.')
    if payload['experiment_id'] != experiment or payload['label_mapping'] != {str(i): label for i, label in enumerate(LABELS)}:
        raise ValueError('Frozen experiment or label mapping differs. Restore the original manifest.')
    return payload


def verify_files(root, payload, names):
    for name in names:
        path = Path(root) / name
        if not path.is_file():
            raise FileNotFoundError(f'Required frozen artifact missing: {name}. Restore the original checksum-matching file; no alternate checkpoint is used.')
        if payload['files'].get(name) != file_hash(path):
            raise ValueError(f'Frozen runtime checksum failed: {name}. Restore the original checksum-matching file.')


def verify_crf_runtime(root=ROOT):
    envelope(root, CRF_MANIFEST, CRF_DIGEST, CRF_ID)
    # Independently pinned front end: a missing neural package must not disable CRF.
    verify_files(root, {'files': {'src/inference/text_preprocessor.py': '88cb8da3ee051748f04fb24faf3b8a634070abd59d5a26b05032fe1ab74aeb4e', 'src/inference/entity_formatter.py': '9b5986e78379ebd9e95e89ece0ff308617190a46307fe341360ce42da0cb1422', 'src/inference/crf_predictor.py': '28bbb811b530e262476ab3fb4350aa83a6a66092ae4f8327b0c0da08f85d4406', 'src/inference/exporter.py': 'f7c12d854b0847b832d8b91b27631c0a26b1bab883c6b0a2b9bbcc6788fb2b92'}}, ('src/inference/text_preprocessor.py', 'src/inference/entity_formatter.py', 'src/inference/crf_predictor.py', 'src/inference/exporter.py'))
    return verify_classical(root)


def verify_neural_runtime(root=ROOT):
    payload = envelope(root, NEURAL_MANIFEST, NEURAL_DIGEST, NEURAL_ID)
    directory = f'models/bilstm_crf/{NEURAL_ID}'
    names = [f'{directory}/{name}' for name in ('best_model.pt', 'vocabulary.json', 'resolved_config.yaml', 'environment.json')]
    names += list(COMMON_FILES) + ['src/models/bilstm_crf.py', 'src/models/linear_chain_crf.py', 'src/data/bilstm_vocabulary.py']
    verify_files(root, payload, names)
    config = yaml.safe_load((Path(root) / directory / 'resolved_config.yaml').read_text())
    if config != payload['config'] or config['experiment_id'] != NEURAL_ID:
        raise ValueError('Frozen neural configuration differs.')
    if importlib.metadata.version('torch') != payload['environment']['pytorch_version']:
        raise ValueError('PyTorch version differs from the frozen environment. Restore the pinned neural dependencies.')
    vocabulary = read_json(Path(root) / directory / 'vocabulary.json')
    mapping = vocabulary['token2id']
    if (digest(mapping) != vocabulary['vocab_sha256'] or len(mapping) != config['vocab_size']
            or sorted(mapping.values()) != list(range(len(mapping)))
            or mapping.get('<PAD>') != 0 or mapping.get('<UNK>') != 1
            or vocabulary['source_split'] != 'train_clean'):
        raise ValueError('Frozen vocabulary mapping is invalid.')
    return payload
