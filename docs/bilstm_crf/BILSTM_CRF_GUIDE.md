# IndicNewsNER BiLSTM-CRF User & Developer Guide

## Overview
This guide documents how to train, evaluate, and reproduce the PyTorch-based **BiLSTM-CRF** neural sequence tagger for Hindi Named Entity Recognition on Apple Silicon (`mps`) or CPU.

---

## Environment Setup & Requirements

```sh
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Verify dependencies and PyTorch MPS
python -c "import torch; print('PyTorch:', torch.__version__, '| MPS available:', torch.backends.mps.is_available())"

# 3. Run unit and integration tests
pytest
```

---

## Controlled Training Commands

All commands must be executed through the guarded pipeline or training entry points. Only `train_clean` and `validation_clean` are accessible during development.

### 1. Smoke Experiment (1,000 records)
```sh
.venv/bin/python -m src.training.train_bilstm_crf --size 1000
```
* Generates sample manifest `experiments/bilstm_crf/manifests/train_001k_seed42.json`.
* Saves trained weights under `models/bilstm_crf/bilstm_crf_001k_seed42_<hash>/`.
* Writes reports under `reports/bilstm_crf/bilstm_crf_001k_seed42_<hash>/`.

### 2. Pilot A Experiment (10,000 records)
```sh
.venv/bin/python -m src.training.train_bilstm_crf --size 10000
```

### 3. Pilot B Experiment (50,000 records)
```sh
.venv/bin/python -m src.training.train_bilstm_crf --size 50000
```

---

## Guarded Development Entry Point

To run guarded development with runtime auditing:
```sh
.venv/bin/python -m scripts.development --module src.training.train_bilstm_crf --size 1000
```

---

## Generating Comparison Reports

To regenerate the scaling summary tables and comparison against the frozen classical CRF baseline:
```sh
.venv/bin/python -m scripts.compare_bilstm_crf
```
Outputs:
* `reports/bilstm_crf/bilstm_crf_scaling_comparison.csv`
* `reports/bilstm_crf/bilstm_crf_scaling_comparison.md`

---

## Sealed Evaluation Safeguards
* **Strict Boundary:** The evaluator (`src/evaluation/evaluate_bilstm_crf.py`) automatically refuses any split containing `test`.
* **Prohibited Files:** `test_clean.jsonl`, `official_test.jsonl`, and historical test prediction files are forbidden during model selection.
