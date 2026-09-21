import pytest
from src.evaluation.metrics import StrictMetrics


def test_full_match():
    metrics = StrictMetrics()
    metrics.add(["B-PER", "I-PER", "O"], ["B-PER", "I-PER", "O"])
    assert metrics.result()["strict_entity_micro_f1"] == 1
    assert metrics.result()["strict_entity_macro_f1"] == pytest.approx(1 / 3)


def test_wrong_boundary_not_partial_credit():
    metrics = StrictMetrics()
    metrics.add(["B-PER", "I-PER"], ["B-PER", "O"])
    assert metrics.result()["strict_entity_micro_f1"] == 0
    assert metrics.result()["token_accuracy"] == 0.5


def test_wrong_type():
    metrics = StrictMetrics()
    metrics.add(["B-PER"], ["B-ORG"])
    assert metrics.result()["strict_entity_micro_f1"] == 0


def test_micro_counts_across_sentences():
    metrics = StrictMetrics()
    metrics.add(["B-PER"], ["B-PER"])
    metrics.add(["B-LOC"], ["O"])
    result = metrics.result()
    assert result["strict_entity_precision"] == 1
    assert result["strict_entity_recall"] == 0.5
    assert result["strict_entity_micro_f1"] == pytest.approx(2 / 3)


def test_shape_rejected():
    with pytest.raises(ValueError):
        StrictMetrics().add(["O"], [])
