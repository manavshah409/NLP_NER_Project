# IndicNewsNER BiLSTM-CRF Model Card: bilstm_crf_100k_seed42_9af1d47db429

## Summary
- **Model Type:** PyTorch BiLSTM-CRF (1 layer, 128 hidden dim per direction, 100 word embedding dim).
- **Training Sample:** 100,000 clean training records from `naamapadam_hi_crf_v1` (Seed: 42).
- **Vocabulary Size:** 94,405 tokens (built strictly from training sample).
- **Validation UNK Rate:** 2.5372%.
- **Device Used:** mps.
- **Best Epoch:** 9 (Early stopping patience: 3, Reason: early_stopping_patience_reached_at_epoch_12).
- **Total Training Time:** 3949.68 seconds.
- **Model Size:** 36.92 MiB (38,716,725 bytes).

## Validation Strict Metrics (`validation_clean`, 12,896 records)
- **Strict Micro F1:** 0.73818829
- **Strict Macro F1:** 0.73339752
- **Precision:** 0.74983164
- **Recall:** 0.72690101
- **PER F1:** 0.78115319
- **ORG F1:** 0.63522998
- **LOC F1:** 0.78380938
- **Token Accuracy:** 0.93356461
- **Median Latency:** 3.601 ms / sentence
- **P95 Latency:** 4.505 ms / sentence

## Research Policy
Evaluated strictly on `validation_clean`. No test split was accessed.
