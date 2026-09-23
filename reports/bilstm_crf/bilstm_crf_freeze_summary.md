# BiLSTM-CRF Baseline Model Freeze Summary

- **Selected Experiment:** `bilstm_crf_100k_seed42_9af1d47db429`
- **Model Architecture:** PyTorch BiLSTM-CRF (1 layer, 128 hidden dim, 100 emb dim, dropout 0.3)
- **Training Sample:** 100,000 clean records (Seed: 42)
- **Validation Strict Micro F1:** **0.73818829**
- **Validation Strict Macro F1:** **0.73339752**
- **PER F1:** 0.7812 | **ORG F1:** 0.6352 | **LOC F1:** 0.7838
- **Token Accuracy:** 0.9336
- **Total Frozen Files:** 68
- **Sealed Boundary:** Evaluated strictly on `validation_clean`. No test data accessed.
- **No Final Test Evaluation:** Confirmed that `test_clean` and `official_test` have not been evaluated.

All file SHA256 checks and the freeze envelope checksum pass.
