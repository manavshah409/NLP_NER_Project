"""Original and synthetic raw text only; no benchmark examples or split reads."""
import builtins
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from streamlit.testing.v1 import AppTest
from src.inference.bilstm_crf_predictor import BiLSTMCRFPredictor
from src.inference.comparison import ClassicalPredictor, compare_entities
from src.inference.crf_predictor import EXAMPLES
from src.inference.exporter import json_export, csv_export
from src.inference import runtime_verification as verification
from src.inference.text_preprocessor import tokenize


@pytest.fixture(scope='module')
def neural():
    return BiLSTMCRFPredictor(device='cpu')


@pytest.fixture(scope='module')
def classical():
    return ClassicalPredictor()


@pytest.mark.parametrize('family,filename', [('neural', 'best_model.pt'), ('neural', 'vocabulary.json'), ('neural', 'resolved_config.yaml'), ('classical', 'model.crfsuite')])
def test_runtime_tampering_rejected(monkeypatch, family, filename):
    module = verification if family == 'neural' else __import__('src.inference.crf_predictor', fromlist=['file_hash'])
    original = module.file_hash
    monkeypatch.setattr(module, 'file_hash', lambda path: 'bad' if Path(path).name == filename else original(path))
    with pytest.raises(ValueError, match='checksum failed'):
        (verification.verify_neural_runtime if family == 'neural' else verification.verify_crf_runtime)()


def test_missing_neural_model_does_not_substitute(monkeypatch):
    original = Path.is_file
    monkeypatch.setattr(Path, 'is_file', lambda p: False if p.name == 'best_model.pt' else original(p))
    with pytest.raises(FileNotFoundError, match='Restore the original'):
        BiLSTMCRFPredictor()
    verification.verify_crf_runtime()  # Classical baseline remains independently usable.


def test_resealed_manifest_is_rejected(monkeypatch):
    original = verification.read_json
    def altered(path):
        result = original(path)
        if str(path).endswith('bilstm_crf_freeze_manifest.json'):
            result['payload']['seed'] = 99
            result['payload_sha256'] = verification.digest(result['payload'])
        return result
    monkeypatch.setattr(verification, 'read_json', altered)
    with pytest.raises(ValueError, match='manifest checksum'):
        verification.verify_neural_runtime()


@pytest.mark.parametrize('text', list(EXAMPLES.values()) + ['राम राम,\nक़ानून नई  दिल्ली! 😀 ज़ज़अनदेखाटोकन'])
def test_shared_schema_and_exact_offsets(neural, classical, text):
    a, b = classical.predict(text), neural.predict(text)
    required = {'text', 'model_family', 'experiment_id', 'language', 'offset_convention', 'tokens', 'predicted_labels', 'entities', 'diagnostics'}
    for result in (a, b):
        assert required <= result.keys()
        assert result['text'] == text and result['language'] == 'hi'
        assert len(result['predicted_labels']) == len(result['tokens'])
        assert all(text[t['start_char']:t['end_char']] == t['text'] for t in result['tokens'])
        assert all(text[e['start_char']:e['end_char']] == e['text'] for e in result['entities'])
        assert json.loads(json_export(result))['text'] == text
        rows = list(csv.DictReader(io.StringIO(csv_export(result).decode('utf-8-sig'))))
        assert len(rows) == len(result['entities'])
    assert a['diagnostics']['unknown_rate'] is None
    assert all(e['confidence'] is None for e in b['entities'])
    comparison = compare_entities(a, b)
    assert len(comparison['agreed']) + len(comparison['neural_only']) == len(b['entities'])
    assert b['diagnostics']['known_tokens'] + b['diagnostics']['unknown_tokens'] == len(b['tokens'])


def test_repeated_inference_is_deterministic(neural):
    text = EXAMPLES['Multiple entities']
    a, b = neural.predict(text), neural.predict(text)
    assert a['tokens'] == b['tokens'] and a['entities'] == b['entities']
    assert not neural.model.training
    assert not any(p.requires_grad for p in neural.model.parameters())


def test_padding_masks_match_unpadded_decode(neural):
    sequences = [neural.vocabulary.encode([t['text'] for t in tokenize(text)]) for text in [EXAMPLES['Person and place'], 'जयपुर।']]
    with torch.inference_mode():
        padded = neural._decode(sequences)
        individual = [neural._decode([sequence])[0] for sequence in sequences]
    assert padded == individual
    assert list(map(len, padded)) == list(map(len, sequences))


def test_unknown_tokens_and_sentence_boundaries(neural, monkeypatch):
    text = 'ज़ज़अनदेखाटोकन।\nजयपुर'
    monkeypatch.setattr(neural, '_decode', lambda sequences: [[5, 0], [6]])
    result = neural.predict(text)
    assert result['tokens'][0]['token_id'] == 1
    assert result['diagnostics']['unknown_tokens'] >= 1
    assert len(result['entities']) == 1  # Leading I-LOC in the new sentence is ignored.


@pytest.mark.parametrize('text', ['', '  \n', 'अ'*5001, 'अ '*301])
def test_neural_input_limits(neural, text):
    with pytest.raises(ValueError):
        neural.predict(text)


def test_cpu_fallback_when_mps_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    predictor = BiLSTMCRFPredictor()
    assert predictor.device == 'cpu' and 'MPS unavailable' in predictor.device_note


def test_unsupported_mps_fallback_is_narrow():
    assert BiLSTMCRFPredictor._unsupported_mps(RuntimeError('operation not implemented for MPS'))
    assert not BiLSTMCRFPredictor._unsupported_mps(RuntimeError('MPS out of memory'))


def forbid_data_access(monkeypatch):
    original_path, original_builtin = Path.open, builtins.open
    def check(path):
        if isinstance(path, int):
            return
        path = Path(path).resolve()
        root = verification.ROOT
        assert not path.is_relative_to(root / 'data'), f'Dataset access: {path}'
        if path.is_relative_to(root / 'reports'):
            assert path in {root / verification.CRF_MANIFEST, root / verification.NEURAL_MANIFEST}, f'Report access: {path}'
    def path_open(path, *args, **kwargs):
        check(path); return original_path(path, *args, **kwargs)
    def builtin_open(path, *args, **kwargs):
        check(path); return original_builtin(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', path_open)
    monkeypatch.setattr(builtins, 'open', builtin_open)


def test_startup_and_inference_need_no_datasets(monkeypatch):
    forbid_data_access(monkeypatch)
    for predictor in (ClassicalPredictor(), BiLSTMCRFPredictor(device='cpu')):
        assert predictor.predict(EXAMPLES['Person and place'])['entities']


def test_guarded_process_inference():
    code = '''import torch
from src.development_guard import install
install()
from src.inference.comparison import ClassicalPredictor
from src.inference.bilstm_crf_predictor import BiLSTMCRFPredictor
for predictor in (ClassicalPredictor(), BiLSTMCRFPredictor(device='cpu')):
    assert predictor.predict('मीरा शर्मा जयपुर गईं।')['language'] == 'hi'
'''
    run = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr


def test_comparison_rejects_different_inputs(classical, neural):
    with pytest.raises(ValueError, match='identical original text'):
        compare_entities(classical.predict('जयपुर'), neural.predict('मुंबई'))


def test_dual_app_modes_and_stale_results(monkeypatch):
    forbid_data_access(monkeypatch)
    app = AppTest.from_file(verification.ROOT / 'app.py', default_timeout=30).run()
    assert not app.exception and not app.error
    app.radio[0].set_value('Compare Both').run()
    app.selectbox[0].select('Person and place').run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert len(app.session_state['results']) == 2
    assert len(app.get('download_button')) == 5
    assert any('Agreement means' in c.value for c in app.caption)
    app.text_area[0].input('नया पाठ').run()
    assert any('input has changed' in i.value for i in app.info)
    app.radio[0].set_value('BiLSTM-CRF').run()
    assert len(app.get('download_button')) == 0
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert len(app.session_state['results']) == 1
    app.button[1].click().run()
    assert app.text_area[0].value == ''


def test_app_neural_failure_leaves_classical_available(monkeypatch):
    import src.inference.bilstm_crf_predictor as module
    def failed(*args, **kwargs):
        raise ValueError('Frozen runtime checksum failed: best_model.pt')
    monkeypatch.setattr(module.BiLSTMCRFPredictor, 'verify', failed)
    app = AppTest.from_file(verification.ROOT / 'app.py', default_timeout=30).run()
    app.radio[0].set_value('Compare Both').run()
    assert app.error and 'BiLSTM-CRF unavailable' in app.error[0].value
    app.selectbox[0].select('Person and place').run()
    app.button[0].click().run()
    assert list(app.session_state['results']) == ['Classical CRF']
    assert len(app.get('download_button')) == 2
