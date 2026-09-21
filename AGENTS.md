# IndicNewsNER research boundaries

The 100k classical CRF is frozen for Milestone 2C. Do not change its configuration,
sample, binary, frozen source or environment without a separately authorised new
version. Verify `reports/crf/100k_baseline_freeze_manifest.json` before final use.

During later model development or selection, do not read test predictions or
illustrative examples in `reports/crf/100k_final/*_predictions.jsonl` or
`reports/crf/100k_final/restricted_examples/`. Those files are for final reporting
only. Use train_clean and validation_clean for model development. Do not tune any
model from the final test metrics. New final test evaluations require explicit
user authorisation. Never rerun a final evaluator to improve a score.

No 250k/full-data CRF, BiLSTM-CRF, IndicBERT, or English work is authorised as part
of Milestone 2C. Preserve raw and frozen derived data.

Milestone 2D closure: future development must use the process guard described in
`docs/progress/MILESTONE_2D_CLOSURE.md`. Install it before any data access.
Direct historical training/evaluation commands are archival only; do not use them
to bypass the guard. Only train_clean and validation_clean may be mounted in a
future isolated development environment. Keep final aggregate metrics available
for reporting but outside model-selection processes.
