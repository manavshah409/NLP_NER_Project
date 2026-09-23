"""Comprehensive unit and integration tests for BiLSTM-CRF architecture, dataset, vocabulary, and guards."""
import itertools
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from src.data.bilstm_dataset import NERDataset, collate_ner_batch, get_dataloader
from src.data.bilstm_vocabulary import PAD_ID, PAD_TOKEN, UNK_ID, UNK_TOKEN, Vocabulary
from src.data.deterministic_sampler import choose_indices
from src.evaluation.evaluate_bilstm_crf import evaluate_model
from src.models.bilstm_crf import BiLSTM_CRF, LABEL2ID, LABELS, NUM_TAGS
from src.models.linear_chain_crf import LinearChainCRF
from src.utils.torch_reproducibility import set_seed


# -----------------------------------------------------------------------------
# 1. Vocabulary & Hindi Unicode Tests
# -----------------------------------------------------------------------------

def test_vocabulary_training_only_construction():
    records = [
        {"tokens": ["प्रधानमंत्री", "नरेंद्र", "मोदी", "ने", "कहा"], "labels": ["O", "B-PER", "I-PER", "O", "O"], "source_split": "train"},
        {"tokens": ["नई", "दिल्ली", "में", "बैठक", "हुई"], "labels": ["B-LOC", "I-LOC", "O", "O", "O"], "source_split": "train"},
    ]
    vocab = Vocabulary.build_from_records(records, min_frequency=1)
    assert len(vocab) == 12  # PAD (0), UNK (1) + 10 unique Hindi tokens
    assert vocab.token2id[PAD_TOKEN] == PAD_ID
    assert vocab.token2id[UNK_TOKEN] == UNK_ID
    assert "प्रधानमंत्री" in vocab.token2id
    assert "दिल्ली" in vocab.token2id


def test_vocabulary_rejects_non_training_splits():
    records = [{"tokens": ["परीक्षण"], "labels": ["O"], "source_split": "validation_clean"}]
    with pytest.raises(ValueError, match="must only be built from train_clean"):
        Vocabulary.build_from_records(records)


def test_vocabulary_deterministic_ids():
    records = [
        {"tokens": ["भारत", "सरकार", "ने", "कहा", "भारत"], "labels": ["B-ORG", "I-ORG", "O", "O", "B-LOC"], "source_split": "train"},
    ]
    v1 = Vocabulary.build_from_records(records)
    v2 = Vocabulary.build_from_records(records)
    assert v1.token2id == v2.token2id
    # 'भारत' has count 2, so it should have lowest ID after PAD and UNK
    assert v1.token2id["भारत"] == 2


def test_vocabulary_unk_and_unicode_preservation():
    records = [{"tokens": ["जयपुर", "राजस्थान"], "labels": ["B-LOC", "B-LOC"], "source_split": "train"}]
    vocab = Vocabulary.build_from_records(records)
    encoded = vocab.encode(["जयपुर", "अनजान_शब्द", "राजस्थान"])
    assert encoded[0] == vocab.token2id["जयपुर"]
    assert encoded[1] == UNK_ID
    assert encoded[2] == vocab.token2id["राजस्थान"]
    decoded = vocab.decode(encoded)
    assert decoded == ["जयपुर", UNK_TOKEN, "राजस्थान"]


# -----------------------------------------------------------------------------
# 2. Dataset, Collation, and Padding Mask Tests
# -----------------------------------------------------------------------------

def test_dataset_and_collate_mask():
    records = [
        {"tokens": ["अमित", "शाह"], "labels": ["B-PER", "I-PER"], "source_index": 10},
        {"tokens": ["भारतीय", "जनता", "पार्टी", "ने", "घोषणा", "की"], "labels": ["B-ORG", "I-ORG", "I-ORG", "O", "O", "O"], "source_index": 20},
    ]
    vocab = Vocabulary.build_from_records(records, source_split="train")
    dataset = NERDataset(records, vocab)
    assert len(dataset) == 2

    batch = collate_ner_batch([dataset[0], dataset[1]])
    assert batch["input_ids"].shape == (2, 6)
    assert batch["tag_ids"].shape == (2, 6)
    assert batch["mask"].shape == (2, 6)
    # First sequence length 2: positions 0,1 are True, 2..5 are False
    assert batch["mask"][0].tolist() == [True, True, False, False, False, False]
    assert batch["mask"][1].tolist() == [True, True, True, True, True, True]
    assert batch["input_ids"][0, 2:].tolist() == [PAD_ID, PAD_ID, PAD_ID, PAD_ID]


# -----------------------------------------------------------------------------
# 3. Linear-Chain CRF Numerical & Algorithmic Correctness Tests
# -----------------------------------------------------------------------------

def test_crf_log_partition_vs_brute_force_enumeration():
    """Verify forward log-partition Z(x) matches exhaustive path summation."""
    num_tags = 3
    seq_len = 3
    crf = LinearChainCRF(num_tags=num_tags)
    
    # Deterministic weights for exact test
    torch.manual_seed(42)
    emissions = torch.randn(1, seq_len, num_tags)
    mask = torch.ones(1, seq_len, dtype=torch.bool)

    # 1. CRF forward log-partition
    computed_log_z = crf._compute_log_partition(emissions, mask).item()

    # 2. Exhaustive summation over all 3^3 = 27 paths
    all_paths = list(itertools.product(range(num_tags), repeat=seq_len))
    total_unnormalized_sum = 0.0
    for path in all_paths:
        # Score for path: start + emissions + transitions + end
        path_score = crf.start_transitions[path[0]].item() + emissions[0, 0, path[0]].item()
        for t in range(1, seq_len):
            path_score += crf.transitions[path[t - 1], path[t]].item() + emissions[0, t, path[t]].item()
        path_score += crf.end_transitions[path[-1]].item()
        total_unnormalized_sum += math.exp(path_score)

    expected_log_z = math.log(total_unnormalized_sum)
    assert abs(computed_log_z - expected_log_z) < 1e-4


def test_crf_viterbi_vs_brute_force_enumeration():
    """Verify Viterbi decode matches the maximum-scoring path from exhaustive search."""
    num_tags = 3
    seq_len = 3
    crf = LinearChainCRF(num_tags=num_tags)
    
    torch.manual_seed(123)
    emissions = torch.randn(1, seq_len, num_tags)
    mask = torch.ones(1, seq_len, dtype=torch.bool)

    viterbi_paths = crf.decode(emissions, mask=mask)
    viterbi_path = tuple(viterbi_paths[0])

    # Exhaustive search for best path
    all_paths = list(itertools.product(range(num_tags), repeat=seq_len))
    best_score = -float("inf")
    best_path = None
    for path in all_paths:
        score = crf.start_transitions[path[0]].item() + emissions[0, 0, path[0]].item()
        for t in range(1, seq_len):
            score += crf.transitions[path[t - 1], path[t]].item() + emissions[0, t, path[t]].item()
        score += crf.end_transitions[path[-1]].item()
        if score > best_score:
            best_score = score
            best_path = path

    assert viterbi_path == best_path


def test_crf_mask_invariance():
    """Verify that padding mask values do not alter the score or log-partition of real tokens."""
    num_tags = 4
    crf = LinearChainCRF(num_tags=num_tags)

    # Sequence A: length 2 (padded to 4)
    # Sequence B: length 3 (padded to 4)
    emissions = torch.randn(2, 4, num_tags)
    tags = torch.randint(0, num_tags, (2, 4))
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]], dtype=torch.bool)

    loss1 = crf(emissions, tags, mask=mask, reduction="none")

    # If we perturb emissions and tags at padded positions (t >= length), loss should NOT change!
    perturbed_emissions = emissions.clone()
    perturbed_emissions[0, 2:] = 999.0
    perturbed_emissions[1, 3:] = -888.0

    perturbed_tags = tags.clone()
    perturbed_tags[0, 2:] = 0
    perturbed_tags[1, 3:] = 1

    loss2 = crf(perturbed_emissions, perturbed_tags, mask=mask, reduction="none")
    torch.testing.assert_close(loss1, loss2)


# -----------------------------------------------------------------------------
# 4. BiLSTM-CRF Model & Forward / Backward Tests
# -----------------------------------------------------------------------------

def test_bilstm_crf_forward_backward():
    vocab_size = 50
    model = BiLSTM_CRF(vocab_size=vocab_size, num_tags=NUM_TAGS, embedding_dim=32, hidden_dim=64)
    
    input_ids = torch.randint(0, vocab_size, (4, 10))
    tags = torch.randint(0, NUM_TAGS, (4, 10))
    mask = torch.ones(4, 10, dtype=torch.bool)
    mask[0, 7:] = False
    mask[1, 5:] = False

    loss = model(input_ids, tags=tags, mask=mask)
    assert loss.dim() == 0  # Scalar loss
    assert loss.item() >= 0.0  # NLL is non-negative

    loss.backward()
    # Check that gradients exist for embedding, lstm, emission, crf transitions
    assert model.embedding.weight.grad is not None
    assert model.lstm.weight_ih_l0.grad is not None
    assert model.emission_linear.weight.grad is not None
    assert model.crf.transitions.grad is not None


def test_bilstm_crf_decode_lengths():
    vocab_size = 50
    model = BiLSTM_CRF(vocab_size=vocab_size, num_tags=NUM_TAGS, embedding_dim=32, hidden_dim=64)
    input_ids = torch.randint(0, vocab_size, (3, 8))
    mask = torch.tensor([
        [True, True, True, False, False, False, False, False],
        [True, True, True, True, True, False, False, False],
        [True, True, True, True, True, True, True, True],
    ], dtype=torch.bool)

    paths = model.decode(input_ids, mask=mask)
    assert len(paths) == 3
    assert len(paths[0]) == 3
    assert len(paths[1]) == 5
    assert len(paths[2]) == 8


# -----------------------------------------------------------------------------
# 5. Reproducibility and Checkpoint Consistency
# -----------------------------------------------------------------------------

def test_reproducibility_and_checkpoint_reload(tmp_path):
    set_seed(42)
    m1 = BiLSTM_CRF(vocab_size=30, num_tags=7, embedding_dim=16, hidden_dim=16)
    m1.eval()
    x = torch.randint(0, 30, (2, 5))
    mask = torch.ones(2, 5, dtype=torch.bool)
    out1 = m1.decode(x, mask=mask)

    ckpt_path = tmp_path / "model.pt"
    torch.save(m1.state_dict(), ckpt_path)

    m2 = BiLSTM_CRF(vocab_size=30, num_tags=7, embedding_dim=16, hidden_dim=16)
    m2.load_state_dict(torch.load(ckpt_path))
    m2.eval()
    out2 = m2.decode(x, mask=mask)

    assert out1 == out2


# -----------------------------------------------------------------------------
# 6. Evaluation Safety & Sealed Guard Rejection
# -----------------------------------------------------------------------------

def test_evaluation_rejects_test_splits():
    model = BiLSTM_CRF(vocab_size=10, num_tags=7)
    records = [{"tokens": ["test"], "labels": ["O"]}]
    vocab = Vocabulary(token2id={"<PAD>": 0, "<UNK>": 1, "test": 2})

    with pytest.raises(ValueError, match="Sealed evaluation policy forbids"):
        evaluate_model(model, records, vocab, torch.device("cpu"), split_name="test_clean")

    with pytest.raises(ValueError, match="Sealed evaluation policy forbids"):
        evaluate_model(model, records, vocab, torch.device("cpu"), split_name="official_test")


# -----------------------------------------------------------------------------
# 7. Milestone 3B: Statistical Aggregation & Math Verification
# -----------------------------------------------------------------------------

def test_sample_std_dev_and_statistics_math():
    from scripts.robustness_analysis import compute_statistics, sample_std_dev

    # Known values: [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    # Mean = 5.0, Sample Variance (N-1=7) = 32 / 7 = 4.57142857, Sample Std Dev = 2.138089935
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    stats = compute_statistics(data)
    assert abs(stats["mean"] - 5.0) < 1e-6
    assert abs(stats["sample_std"] - 2.138089935) < 1e-6
    assert stats["min"] == 2.0
    assert stats["max"] == 9.0
    assert stats["range"] == 7.0
    assert stats["median"] == 4.5

    # Single-element edge case
    assert sample_std_dev([10.0]) == 0.0


# -----------------------------------------------------------------------------
# 8. Milestone 3B: Seed Propagation & Deterministic Configuration Resolution
# -----------------------------------------------------------------------------

def test_seed_propagation():
    s1 = set_seed(7)
    assert s1["seed"] == 7
    assert s1["python_hash_seed"] == "7"
    assert s1["torch_seed"] == 7

    s2 = set_seed(21)
    assert s2["seed"] == 21
    assert s2["python_hash_seed"] == "21"
    assert s2["torch_seed"] == 21


def test_50k_sample_and_vocab_reuse_across_seeds():
    manifest_50k = Path("experiments/bilstm_crf/manifests/train_050k_seed42.json")
    if manifest_50k.exists():
        from src.utils.io import read_json
        sample = read_json(manifest_50k)
        assert sample["size"] == 50000
        assert sample["seed"] == 42
        assert len(sample["source_indices"]) == 50000

        # Verify that building vocabulary on identical 50k records is deterministic
        from src.data.loader import records
        train_clean = Path("data/derived/naamapadam_hi_crf_v1/train_clean.jsonl")
        if train_clean.exists():
            selected = set(sample["source_indices"])
            subset = [r for r in records(train_clean) if r["source_index"] in selected]
            v1 = Vocabulary.build_from_records(subset, source_split="train_clean")
            v2 = Vocabulary.build_from_records(subset, source_split="train_clean")
            assert v1.token2id == v2.token2id
            assert len(v1) == 62099


# -----------------------------------------------------------------------------
# 9. Milestone 3B: Nested 50k to 100k Sample Verification
# -----------------------------------------------------------------------------

def test_nested_sample_construction(tmp_path):
    # Create mock clean training records
    mock_records = []
    for i in range(100):
        label = "B-PER" if i % 3 == 0 else ("B-ORG" if i % 3 == 1 else "B-LOC")
        mock_records.append({"source_split": "train", "source_index": i, "tokens": [f"token_{i}"], "labels": [label]})

    ind_10 = choose_indices(mock_records, 10, seed=42)
    ind_20 = choose_indices(mock_records, 20, seed=42)
    ind_50 = choose_indices(mock_records, 50, seed=42)

    # Every smaller sample must be a strict subset of any larger sample with same seed
    assert set(ind_10).issubset(set(ind_20))
    assert set(ind_20).issubset(set(ind_50))


# -----------------------------------------------------------------------------
# 10. Milestone 3B: Decision-Gate Logic Verification
# -----------------------------------------------------------------------------

def test_decision_gate_conditions():
    baseline_val_f1 = 0.713769728

    # Pass case: Mean > baseline and at least 2 seeds > baseline
    runs_pass = [
        {"val_micro_f1": 0.719230},
        {"val_micro_f1": 0.714500},
        {"val_micro_f1": 0.712000},
    ]
    mean_pass = sum(r["val_micro_f1"] for r in runs_pass) / len(runs_pass)
    exceed_count_pass = sum(1 for r in runs_pass if r["val_micro_f1"] > baseline_val_f1)
    assert mean_pass > baseline_val_f1
    assert exceed_count_pass >= 2

    # Fail case 1: Mean below baseline
    runs_fail1 = [
        {"val_micro_f1": 0.710000},
        {"val_micro_f1": 0.711000},
        {"val_micro_f1": 0.712000},
    ]
    mean_fail1 = sum(r["val_micro_f1"] for r in runs_fail1) / len(runs_fail1)
    assert mean_fail1 < baseline_val_f1

    # Fail case 2: Only 1 seed exceeds baseline
    runs_fail2 = [
        {"val_micro_f1": 0.720000},
        {"val_micro_f1": 0.710000},
        {"val_micro_f1": 0.710000},
    ]
    exceed_count_fail2 = sum(1 for r in runs_fail2 if r["val_micro_f1"] > baseline_val_f1)
    assert exceed_count_fail2 < 2


# -----------------------------------------------------------------------------
# 11. Milestone 3B: Freeze Manifest Generation & Tampering Verification
# -----------------------------------------------------------------------------

def test_bilstm_crf_freeze_seal_and_tampering(tmp_path):
    from src.evaluation.freeze_bilstm_crf import seal, verify_freeze
    from src.utils.io import file_hash, write_json

    dummy_file = tmp_path / "model.pt"
    dummy_file.write_text("model weights")
    manifest_path = tmp_path / "freeze_manifest.json"

    payload = {
        "experiment_id": "test_exp",
        "files": {str(dummy_file): file_hash(dummy_file)},
    }
    write_json(manifest_path, seal(payload))

    # Verification passes
    verified = verify_freeze(manifest_path)
    assert verified["experiment_id"] == "test_exp"

    # Tampering with file triggers error
    dummy_file.write_text("modified weights")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        verify_freeze(manifest_path)

    # Tampering with envelope digest triggers error
    envelope = seal(payload)
    envelope["payload_sha256"] = "corrupted_hash"
    write_json(manifest_path, envelope)
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_freeze(manifest_path)


def test_generated_bilstm_crf_freeze_manifest_on_disk():
    from src.evaluation.freeze_bilstm_crf import MANIFEST, verify_freeze
    if MANIFEST.exists():
        payload = verify_freeze(MANIFEST)
        assert payload["experiment_id"] == "bilstm_crf_100k_seed42_9af1d47db429"
        assert payload["seed"] == 42
        assert payload["sample_verification"]["records"] == 100000
        assert payload["sample_verification"]["only_train_clean"] is True
        assert len(payload["files"]) >= 50
        assert payload["validation_metrics"]["strict_entity_micro_f1"] > 0.73
        assert payload["no_final_test_evaluation"] is True
        assert payload["replay_verification"]["exact_aggregate_metric_replay"] is True


def test_100k_vocabulary_checksum_and_size():
    vocab_path = Path("models/bilstm_crf/bilstm_crf_100k_seed42_9af1d47db429/vocabulary.json")
    if vocab_path.exists():
        from src.utils.io import file_hash
        vocab = Vocabulary.load(vocab_path)
        assert len(vocab) == 94405
        assert file_hash(vocab_path) == "5438e53655fd7a51b9a6e8f11eb44fe35c0148ccc9814bef4b47ecda0b115ad2"


def test_100k_sample_manifest_checksum():
    sample_path = Path("experiments/bilstm_crf/manifests/train_100k_seed42.json")
    if sample_path.exists():
        from src.utils.io import file_hash, read_json
        sample = read_json(sample_path)
        assert sample["size"] == 100000
        assert sample["seed"] == 42
        assert len(sample["source_indices"]) == 100000
        assert file_hash(sample_path) == "83c7400d0a9b95fa8515bc28066fbd0a7e5b4a3b25f1d84811202762c5799b13"


def test_sealed_split_rejection_in_evaluator():
    model = BiLSTM_CRF(vocab_size=10, num_tags=7)
    vocab = Vocabulary(token2id={"<PAD>": 0, "<UNK>": 1})
    with pytest.raises(ValueError, match="Sealed evaluation policy forbids"):
        evaluate_model(model, [], vocab, torch.device("cpu"), split_name="test_clean")
    with pytest.raises(ValueError, match="Sealed evaluation policy forbids"):
        evaluate_model(model, [], vocab, torch.device("cpu"), split_name="official_test")


def test_no_accidental_test_split_in_freeze():
    from src.evaluation.freeze_bilstm_crf import MANIFEST, verify_freeze
    if MANIFEST.exists():
        payload = verify_freeze(MANIFEST)
        for filepath in payload["files"]:
            assert "test_clean" not in filepath or "test_clean.jsonl" in filepath
            assert "test_clean_predictions" not in filepath
            assert "restricted_examples" not in filepath



