from copy import deepcopy
import pytest
from src.data.bio import validate, normalise, strict_spans
from src.data.derive import clean_split
from src.data.loader import normalised_text
from src.data.alignment import align_labels
from src.data.audit import Statistics
from src.utils.io import write_json, read_json


@pytest.mark.parametrize("labels,category", [(["I-PER"], "orphan_I"), (["O", "I-LOC"], "orphan_I"), (["B-PER", "I-ORG"], "type_mismatch_I"), (["Z"], "unknown_label")])
def test_bio_errors(labels, category):
    assert validate(["अ"] * len(labels), labels)[0]["category"] == category


@pytest.mark.parametrize("tokens,labels,category", [(None, [], "missing_or_null_fields"), ([], [], "empty_record"), (["अ"], [], "length_mismatch"), ([""], ["O"], "invalid_token")])
def test_structural_errors(tokens, labels, category):
    assert validate(tokens, labels)[0]["category"] == category


def test_normalisation_manifest_nonmutation():
    tokens, labels = ["राम", "बैंक", "दिल्ली"], ["I-PER", "I-ORG", "I-LOC"]
    original = deepcopy((tokens, labels))
    t, y, repairs = normalise(tokens, labels)
    assert (tokens, labels) == original
    assert t is not tokens and y is not labels
    assert y == ["B-PER", "B-ORG", "B-LOC"]
    assert repairs[1] == {"token_position": 1, "token_text": "बैंक", "original_label": "I-ORG", "derived_label": "B-ORG", "error_category": "type_mismatch_I"}
    assert validate(t, y) == []


def test_strict_raw_spans():
    assert strict_spans(["I-PER", "I-PER", "B-ORG", "I-ORG", "I-LOC"]) == [(2, 4, "ORG")]


def test_valid_normalisation_unchanged():
    assert normalise(["नई", "दिल्ली"], ["B-LOC", "I-LOC"])[1:] == (["B-LOC", "I-LOC"], [])


def row(i, word, tag="B-PER", split="train"):
    return {"source_index": i, "source_split": split, "tokens": [word], "labels": [tag]}


def run_clean(rows, split="train", earlier=None):
    repairs, removed = [], []
    kept = list(clean_split(rows, split, earlier or {}, repairs.append, lambda k, r: removed.append((k, r))))
    return kept, repairs, removed


def test_duplicate_retention_after_repair():
    kept, repairs, removed = run_clean([row(0, "राम", "I-PER"), row(1, "राम")])
    assert [r["source_index"] for r in kept] == [0]
    assert len(repairs) == 1
    assert removed[0][1]["retained_index"] == 0


def test_leakage_before_dedup():
    kept, _, removed = run_clean([row(0, "राम"), row(1, "राम"), row(2, "सीता")], "validation", {"train": {"राम"}})
    assert [r["source_index"] for r in kept] == [2]
    assert [kind for kind, _ in removed] == ["leakage", "leakage"]


def test_empty_test_quarantine():
    empty = {"tokens": [], "labels": [], "source_index": 0, "source_split": "test"}
    kept, _, removed = run_clean([empty], "test")
    assert not kept and removed[0][0] == "quarantine"


def test_unexpected_empty_train_rejected():
    with pytest.raises(ValueError):
        run_clean([{"tokens": [], "labels": [], "source_index": 0}])


def test_derived_reproducibility():
    source = [row(0, "राम"), row(1, "राम"), row(2, "सीता")]
    assert run_clean(source) == run_clean(source)


def test_normalised_text_unicode_and_case():
    assert normalised_text(["नई", " दिल्ली "]) == "नई दिल्ली"
    assert normalised_text(["ABC"]) != normalised_text(["abc"])


def test_alignment_first_only():
    assert align_labels(["B-PER", "O"], [None, 0, 0, 1, None]) == [-100, 1, -100, 0, -100]


def test_alignment_all_subtokens():
    assert align_labels(["B-LOC"], [None, 0, 0], True) == [-100, 5, 6]


def test_alignment_out_of_bounds():
    with pytest.raises(ValueError):
        align_labels(["O"], [1])


def test_statistics_survive_manifest_roundtrip(tmp_path):
    stats = Statistics()
    stats.add(row(0, "राम"))
    path = tmp_path / "stats.json"
    write_json(path, stats.result())
    assert stats.result() == read_json(path)
