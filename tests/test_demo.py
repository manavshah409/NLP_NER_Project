import csv
import builtins
import io
import json
from html.parser import HTMLParser
from pathlib import Path
import pytest
from src.inference.crf_predictor import CRFPredictor, EXAMPLES, EXPERIMENT, ROOT
from src.inference.entity_formatter import format_entities, highlight, mean_marginal
from src.inference.exporter import json_export, csv_export
from src.inference.text_preprocessor import tokenize, sentence_ranges, MAX_CHARACTERS


@pytest.mark.parametrize("text", ["", "  \n\t", "अ" * (MAX_CHARACTERS+1)])
def test_invalid_input(text):
    with pytest.raises(ValueError):
        tokenize(text)


def test_unicode_token_offsets():
    text = "  नई दिल्ली,\nक़ानून और हिंदी।  "
    tokens = tokenize(text)
    assert [t["text"] for t in tokens] == ["नई", "दिल्ली", ",", "क़ानून", "और", "हिंदी", "।"]
    assert all(text[t["start_char"]:t["end_char"]] == t["text"] for t in tokens)
    assert list(sentence_ranges(text, tokens)) == [(0, 3), (3, 7)]


def test_multiple_repeated_adjacent_entities():
    text = "राम राम दिल्ली"
    entities = format_entities(text, tokenize(text), ["B-PER", "B-PER", "B-LOC"], [0.9, 0.8, 0.7])
    assert [e["text"] for e in entities] == ["राम", "राम", "दिल्ली"]
    assert [(e["start_token"], e["end_token"]) for e in entities] == [(0,1),(1,2),(2,3)]
    assert entities[0]["start_char"] != entities[1]["start_char"]
    assert all(text[e["start_char"]:e["end_char"]] == e["text"] for e in entities)
    adjacent = format_entities("राम,", tokenize("राम,"), ["B-PER","B-ORG"], [0.9,0.8])
    assert adjacent[0]["end_char"] == adjacent[1]["start_char"]
    assert highlight("राम,", adjacent).count("<mark ") == 2


def test_invalid_i_does_not_start_span():
    entities = format_entities("राम दिल्ली", tokenize("राम दिल्ली"), ["I-PER", "B-LOC"], [0.5,0.9])
    assert [e["text"] for e in entities] == ["दिल्ली"]


def test_substring_includes_original_whitespace():
    text = "नई  दिल्ली"
    e = format_entities(text, tokenize(text), ["B-LOC","I-LOC"], [0.8,0.6])[0]
    assert e["text"] == text and e["end_char"] == len(text)
    assert e["confidence"] == pytest.approx(0.7)


@pytest.mark.parametrize("values", [[], [-0.1], [1.1], [float("nan")]])
def test_bad_marginals(values):
    with pytest.raises(ValueError):
        mean_marginal(values)


def test_html_escaping_and_preservation():
    text = '<script>alert("x")</script>\nराम & राम'
    tokens = tokenize(text)
    labels = ["B-PER" if t["text"] == "राम" else "O" for t in tokens]
    markup = highlight(text, format_entities(text,tokens,labels,[0.8]*len(tokens)))
    assert '<script>' not in markup and '&lt;script&gt;' in markup
    class Parser(HTMLParser):
        def __init__(self):
            super().__init__(); self.text = []
        def handle_data(self, data):
            self.text.append(data)
    parser=Parser(); parser.feed(markup)
    assert ''.join(parser.text) == text


def test_json_csv_schema():
    text = "राम दिल्ली"
    result = {"text":text,"entities":format_entities(text,tokenize(text),["B-PER","B-LOC"],[0.8,0.9])}
    assert json.loads(json_export(result))["text"] == text
    rows = list(csv.DictReader(io.StringIO(csv_export(result).decode("utf-8-sig"))))
    assert len(rows)==2 and rows[0]["text"]=="राम" and rows[1]["type"]=="LOC"
    assert rows[0]["start_char"]=="0"


def test_csv_formula_safety():
    e={"text":"=1+1","type":"ORG","start_char":0,"end_char":4,"start_token":0,"end_token":1,"bio_sequence":["B-ORG"],"confidence":0.5}
    rows=list(csv.DictReader(io.StringIO(csv_export({"entities":[e]}).decode("utf-8-sig"))))
    assert rows[0]["text"]=="'=1+1"


def test_checksum_failure(monkeypatch):
    import src.inference.crf_predictor as module
    original=module.file_hash
    monkeypatch.setattr(module,"file_hash",lambda p: "wrong" if str(p).endswith("model.crfsuite") else original(p))
    with pytest.raises(ValueError,match="checksum failed"):
        CRFPredictor()


def test_inference_without_dataset_access(monkeypatch):
    original=Path.open
    original_builtin=builtins.open
    def check_path(path):
        if isinstance(path, int):
            return
        resolved=Path(path).resolve()
        assert not resolved.is_relative_to(ROOT / "data")
        assert not any(part in str(resolved) for part in ("test_clean_predictions","official_test_predictions","restricted_examples"))
    def guarded(path,*args,**kwargs):
        check_path(path)
        return original(path,*args,**kwargs)
    def guarded_builtin(path,*args,**kwargs):
        check_path(path)
        return original_builtin(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    monkeypatch.setattr(builtins,"open",guarded_builtin)
    model=CRFPredictor()
    text=EXAMPLES["Person and place"]
    first=model.predict(text); second=model.predict(text)
    assert first["model"]==EXPERIMENT and first["text"]==text
    assert {"dataset_version","processed_at","entities","offset_convention","limitations","predicted_bio_sequence"} <= first.keys()
    assert first["entities"]==second["entities"] and first["tokens"]==second["tokens"]
    assert all(0<=e["confidence"]<=1 for e in first["entities"])
    assert first["entities"]


def test_long_sentence_rejected():
    text="अ "*301
    with pytest.raises(ValueError,match="sentence exceeds"):
        list(sentence_ranges(text,tokenize(text)))
