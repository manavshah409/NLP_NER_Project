from copy import deepcopy
from pathlib import Path
import sys
import pytest
import yaml
from src.data.crf_features import token_features, sentence_features
from src.data.deterministic_sampler import choose_indices
from src.models.crf_model import CRFModel
from src.utils.resource_monitor import exceeded, monitor
from src.utils.io import read_json
from src.evaluation.evaluate_crf import evaluate

CONFIG = yaml.safe_load(Path("configs/crf_baseline.yaml").read_text())


def test_unicode_features_nonmutation():
    tokens = ["नई", "दिल्ली", "ABC", "ABCदिल्ली"]
    before = deepcopy(tokens)
    result = sentence_features(tokens, CONFIG["features"])
    assert tokens == before
    assert result[0]["token"] == "नई"
    assert "latin_lower" not in result[0]
    assert result[2]["latin_lower"] == "abc"
    assert "latin_lower" not in result[3]
    assert result[3]["mixed_script"]


@pytest.mark.parametrize("token,key", [("राम", "devanagari"), ("BANK", "latin"), ("Bankबैंक", "mixed_script"), ("१२३", "numeric"), ("।", "punctuation"), ("राम,", "contains_punctuation")])
def test_token_properties(token, key):
    assert token_features(token, CONFIG["features"])[key]


def test_context_boundaries():
    features = sentence_features(["राम", "आए"], CONFIG["features"])
    assert features[0]["BOS"] and not features[0]["EOS"]
    assert features[1]["EOS"] and not features[1]["BOS"]
    assert "+1:token" in features[0] and "-1:token" not in features[0]
    assert "-1:token" in features[1] and "+1:token" not in features[1]


def test_context_zero():
    config = dict(CONFIG["features"], context_window=0)
    assert "+1:token" not in sentence_features(["राम", "आए"], config)[0]


def test_empty_token_rejected():
    with pytest.raises(ValueError):
        sentence_features([""], CONFIG["features"])


def sample_rows():
    return [{"source_split": "train", "source_index": i, "tokens": [str(i)], "labels": ["B-" + ("PER", "ORG", "LOC")[i % 3]]} for i in range(100)]


def test_sampling_seed_and_nesting():
    rows = sample_rows()
    a = choose_indices(rows, 10, 42)
    assert a == choose_indices(rows, 10, 42)
    assert a != choose_indices(rows, 10, 43)
    assert set(a) <= set(choose_indices(rows, 50, 42))
    assert {rows[i]["labels"][0] for i in a} == {"B-PER", "B-ORG", "B-LOC"}


def test_sampling_validation_rejected():
    rows = sample_rows()
    rows[0]["source_split"] = "validation"
    with pytest.raises(ValueError):
        choose_indices(rows, 10, 42)


def test_fit_predict_native_reload(tmp_path):
    tokens = ["राम", "दिल्ली"]
    features = sentence_features(tokens, CONFIG["features"])
    model = CRFModel(dict(CONFIG["model"], max_iterations=10))
    trainer = model.trainer()
    for _ in range(5):
        trainer.append(features, ["B-PER", "B-LOC"])
    path = tmp_path / "model.crfsuite"
    trainer.train(str(path))
    predicted = model.load(path).predict(features)
    assert predicted == ["B-PER", "B-LOC"]
    assert CRFModel(CONFIG["model"]).load(path).predict(features) == predicted
    assert len(predicted) == len(tokens)


def test_memory_threshold():
    policy = {"maximum_memory_bytes": 1000, "minimum_available_bytes": 100}
    assert exceeded(900, 1000, policy)
    assert exceeded(100, 99, policy)
    assert not exceeded(800, 101, policy)


def test_monitor_simulated_interruption(tmp_path):
    policy = {"maximum_memory_bytes": 1, "minimum_available_bytes": 0}
    output = tmp_path / "resources.json"
    with pytest.raises(RuntimeError, match="memory_limit=True"):
        monitor([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path / "log", output, policy, poll=0.01)
    assert read_json(output)["memory_limit_interruption"]


def test_test_evaluation_rejected(tmp_path):
    with pytest.raises(ValueError, match="Only validation_clean"):
        evaluate("unused", "official_test.jsonl", CONFIG, set(), tmp_path)
