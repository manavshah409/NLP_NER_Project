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
# Baseline seed 42
.venv/bin/python -m src.training.train_bilstm_crf --size 50000 --seed 42

# Robustness repeated seeds (same 50k sample and vocabulary)
.venv/bin/python -m src.training.train_bilstm_crf --size 50000 --seed 7
.venv/bin/python -m src.training.train_bilstm_crf --size 50000 --seed 21
```

### 4. Controlled 100k Experiment
```sh
.venv/bin/python -m src.training.train_bilstm_crf --size 100000 --seed 42
```
* Generates nested sample manifest `experiments/bilstm_crf/manifests/train_100k_seed42.json`.
* Builds 94,405-token training vocabulary.
* Saves model under `models/bilstm_crf/bilstm_crf_100k_seed42_<hash>/`.

---

## Robustness & Final Model Selection

To compute 3-seed robustness metrics and generate comparison reports:
```sh
# 1. Generate 50k robustness analysis
.venv/bin/python -m scripts.robustness_analysis

# 2. Generate final model comparison and selection report
.venv/bin/python -m scripts.selection
```
Outputs:
* `reports/bilstm_crf/robustness_050k.csv` & `robustness_050k.md`
* `reports/bilstm_crf/bilstm_crf_final_comparison.csv` & `BILSTM_CRF_FINAL_SELECTION.md`

---

## Freezing the Baseline Model

To freeze the approved BiLSTM-CRF baseline model package:
```sh
.venv/bin/python -m src.evaluation.freeze_bilstm_crf --experiment-id bilstm_crf_100k_seed42_9af1d47db429
```
Outputs:
* `reports/bilstm_crf/bilstm_crf_freeze_manifest.json`
* `reports/bilstm_crf/bilstm_crf_freeze_summary.md`

---

## Guarded Development Entry Point

To run guarded development with runtime auditing:
```sh
.venv/bin/python -m scripts.development --module src.training.train_bilstm_crf --size 1000
```

---

## Sealed Evaluation Safeguards
* **Strict Boundary:** The evaluator (`src/evaluation/evaluate_bilstm_crf.py`) automatically refuses any split containing `test`.
* **Prohibited Files:** `test_clean.jsonl`, `official_test.jsonl`, and historical test prediction files are forbidden during model selection.
