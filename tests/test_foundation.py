"""Synthetic fixtures test infrastructure; these are not dataset observations."""
import pytest
from src.data.acquire import discover_labels
from src.utils.io import checksums, verify_checksums, write_json, read_json
from src.utils.environment import environment


LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def test_schema_discovery():
    assert discover_labels(f"x = datasets.ClassLabel(names={LABELS!r})") == LABELS


def test_schema_is_not_executed(tmp_path):
    marker = tmp_path / "executed"
    source = f"open({str(marker)!r}, 'w').write('bad')\nx = datasets.ClassLabel(names={LABELS!r})"
    assert discover_labels(source) == LABELS
    assert not marker.exists()


@pytest.mark.parametrize("source", ["x = 1", "x = datasets.ClassLabel(names=['O'])", "x = datasets.ClassLabel(names=unknown)"])
def test_invalid_schema(source):
    with pytest.raises(ValueError):
        discover_labels(source)


def test_checksum_stable_and_detects_change(tmp_path):
    (tmp_path / "a").write_text("हिंदी", encoding="utf-8")
    first = checksums(tmp_path)
    assert verify_checksums(tmp_path, first) == first
    (tmp_path / "a").write_text("changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_checksums(tmp_path, first)


def test_checksum_detects_new_file(tmp_path):
    (tmp_path / "a").write_text("1")
    first = checksums(tmp_path)
    (tmp_path / "b").write_text("2")
    with pytest.raises(ValueError):
        verify_checksums(tmp_path, first)


def test_checksum_relative_paths(tmp_path):
    for folder in ("one", "two"):
        root = tmp_path / folder
        root.mkdir()
        (root / "file").write_text("same")
    assert checksums(tmp_path / "one") == checksums(tmp_path / "two")


def test_empty_checksum_rejected(tmp_path):
    with pytest.raises(ValueError):
        checksums(tmp_path)


def test_json_unicode_roundtrip(tmp_path):
    path = tmp_path / "result.json"
    value = {"tokens": ["भारतीय", "बैंक"]}
    write_json(path, value)
    assert read_json(path) == value
    assert "भारतीय" in path.read_text()


def test_environment_reports_observations():
    report = environment()
    assert report["memory_total_bytes"] > 0
    assert report["memory_available_bytes"] > 0
    assert report["packages"]["pytest"]
    assert report["seed"] == 42
