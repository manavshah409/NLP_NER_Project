"""Language tools use only authored text; no project benchmark data."""
import json
from pathlib import Path
import subprocess
import pytest
from streamlit.testing.v1 import AppTest
from src.inference import language_bridge as bridge
from src.language_tools import worker


@pytest.mark.parametrize('task,text', [('bad', 'hello'), ('english_ner', ''), ('translate_hi_en', ' '*3), ('english_ner', 'x'*5001)])
def test_bridge_rejects_invalid_requests(task, text):
    with pytest.raises(ValueError):
        bridge.run_language_task(task, text)


def test_missing_environment_is_actionable(monkeypatch):
    monkeypatch.setattr(Path, 'is_file', lambda _: False)
    with pytest.raises(RuntimeError, match='Language environment missing'):
        bridge.run_language_task('english_ner', 'hello')


def test_worker_uses_offline_pipes_and_hides_stderr(monkeypatch):
    monkeypatch.setattr(Path, 'is_file', lambda _: True)
    def run(command, **kwargs):
        assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        assert json.loads(kwargs['input'])['text'] == 'private text'
        assert 'private text' not in command
        return subprocess.CompletedProcess(command, 1, '', 'private text')
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(RuntimeError) as error:
        bridge.run_language_task('english_ner', 'private text')
    assert 'private text' not in str(error.value)


def test_timeout_is_actionable(monkeypatch):
    monkeypatch.setattr(Path, 'is_file', lambda _: True)
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 120)
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(RuntimeError, match='120 seconds'):
        bridge.run_language_task('english_ner', 'hello')


def test_language_model_tampering_rejected(monkeypatch):
    manifest = json.loads((bridge.ROOT / 'configs/language_tools_manifest.json').read_text())
    monkeypatch.setattr(worker.importlib.metadata, 'version', lambda name: manifest['packages'][name])
    monkeypatch.setattr(Path, 'is_file', lambda _: False)
    with pytest.raises(ValueError, match='checksum failed'):
        worker.verify_artifacts('translate_hi_en')


def test_translation_ui_and_stale_output(monkeypatch):
    def fake(task, text):
        return {'text': text, 'translated_text': 'Synthetic test output', 'limitations': 'Unverified translation',
                'elapsed_ms_including_model_load': 1}
    monkeypatch.setattr(bridge, 'run_language_task', fake)
    app = AppTest.from_file(bridge.ROOT / 'pages/1_English_and_Translation.py').run()
    app.radio[0].set_value('Hindi → English translation').run()
    app.button[0].click().run()
    app.button[1].click().run()
    assert not app.exception and not app.error
    assert len(app.get('download_button')) == 2
    assert any('change names' in w.value for w in app.warning)
    app.text_area[0].input('बदला हुआ पाठ').run()
    assert len(app.get('download_button')) == 0
    assert any('changed' in i.value for i in app.info)


def test_english_ui_exports_source_labels_without_csv_schema_error(monkeypatch):
    def fake(task, text):
        return {'text': text, 'entities': [{'text': 'London', 'type': 'LOC', 'source_label': 'GPE',
                'start_char': 0, 'end_char': 6, 'start_token': 0, 'end_token': 1,
                'bio_sequence': ['B-LOC'], 'confidence': None}],
                'limitations': 'Unverified English baseline', 'elapsed_ms_including_model_load': 1}
    monkeypatch.setattr(bridge, 'run_language_task', fake)
    app = AppTest.from_file(bridge.ROOT / 'pages/1_English_and_Translation.py').run()
    app.text_area[0].input('London').run()
    app.button[1].click().run()
    assert not app.exception and len(app.get('download_button')) == 2


@pytest.mark.skipif(not (bridge.ROOT / '.venv-language/bin/python').exists(), reason='Optional isolated language environment absent')
def test_real_offline_language_smoke():
    text = 'Meera Sharma visited London and met the Microsoft team.\nLondon again.'
    result = bridge.run_language_task('english_ner', text)
    assert result['language'] == 'en' and result['text'] == text
    assert {'PER', 'ORG', 'LOC'} <= {e['type'] for e in result['entities']}
    assert all(text[e['start_char']:e['end_char']] == e['text'] for e in result['entities'])
    assert all(text[t['start_char']:t['end_char']] == t['text'] for t in result['tokens'])
    translated = bridge.run_language_task('translate_hi_en', 'मीरा शर्मा ने जयपुर में पुस्तक प्रदर्शनी देखी।')
    assert translated['target_language'] == 'en' and translated['translated_text']
    assert translated['alignment'] is None  # Smoke test, not translation quality certification.
    with pytest.raises(ValueError, match='512 model tokens'):
        bridge.run_language_task('translate_hi_en', 'नमस्ते ' * 650)
