# Milestone 3C — dual-model Hindi demonstration

Completed locally on 2026-10-07. The app supports Classical CRF, BiLSTM-CRF and
Compare Both. Neither frozen model was retrained, tuned or replaced. The same
Unicode-preserving tokenizer and sentence segmentation feed both adapters.

## Baselines and evidence

| Validation metric | Classical CRF | BiLSTM-CRF |
|---|---:|---:|
| Strict micro F1 | 0.713769728 | 0.738188290 |
| Strict macro F1 | 0.709317452 | 0.733397520 |
| PER F1 | 0.751631012 | 0.781153190 |
| ORG F1 | 0.608512135 | 0.635229980 |
| LOC F1 | 0.767809209 | 0.783809380 |

Both use 100,000 training records. Classical experiment:
`crf_100k_seed42_24dd05e84a26`; neural experiment:
`bilstm_crf_100k_seed42_9af1d47db429`. Neural vocabulary: 94,405 entries;
validation UNK rate: 2.54%. These are offline validation results, not current-news
accuracy. The classical historical clean-test result belongs to a separate final
evaluation. No neural final test was run and no cross-split score comparison is valid.

The pinned freeze envelopes and all non-dataset files were verified: 38 classical
and 55 neural entries. Dataset entries were deliberately not opened or rehashed.
Runtime verification checks the exact model, configuration, vocabulary, mapping,
relevant frozen source and required dependency versions. No alternate checkpoint
is loaded on failure. Newly added adapters do not modify frozen source files.

## Application behavior

The neural adapter reconstructs the frozen architecture, loads weights strictly,
uses evaluation mode without gradients, pads bounded batches with masks, maps
unseen words to UNK and decodes with the trained CRF. MPS is preferred; CPU uses
the same checkpoint when MPS is unavailable or lacks a required operation.
Other failures are surfaced. Models are cached as Streamlit resources and their
artifacts are checked on each use. MPS inference was observed in the live browser;
CPU inference is covered by tests. Cross-device bitwise identity is not claimed.

Input is limited to 5,000 Unicode code points and 300 tokens per sentence. Text is
not normalized. Tokens and entities retain zero-based, end-exclusive offsets into
the exact original input. Each entity slice is verified. Empty input is rejected.
Repeated mentions remain distinct; invalid leading I-tags do not start entities.
The neural representation is exact vocabulary IDs; the classical representation
is hand-built features over the same tokens. New text and mode changes invalidate
displayed results. Verification failures hide results for the affected model.

Output includes text, model_family, experiment_id, language, offset_convention,
tokens, predicted_labels, entities and diagnostics. Classical vocabulary statistics
are null because a fixed word vocabulary does not apply. Neural confidence is null;
classical mean marginals are not calibrated correctness. Timing includes checks
and processing, excludes model load, and is not a controlled speed benchmark.
Per-model JSON/CSV and combined comparison JSON are available. Agreement means
identical span and type, not correctness. User input remains in session memory;
only explicit downloads persist it. CSV formula protection may prefix an apostrophe.

## Evaluation boundary incident and correction

During the earlier initial baseline suite, two inherited freeze tests opened sealed
dataset files to hash their bytes. They did not decode examples or run evaluation,
but this still violated the no-access boundary. Those tests now verify non-dataset
artifacts only. An autouse test guard blocks real datasets and record-level reports;
the 50k vocabulary test compares saved artifacts instead of reopening training rows.
All tests run in this closure used that protection. The existing process guard
remains required for future development; it is an accidental-access safeguard,
not an operating-system security boundary.

## Verification and limitations

The 159-test Hindi suite passed before language-tool integration. The final expanded
suite result is recorded in `MILESTONE_3C_VERIFICATION.md`. Browser checks confirmed
both model columns, MPS, highlighted entities, tables and comparison JSON download;
the downloaded JSON was parsed and its entity slices verified. Original authored
examples were used, never restricted benchmark examples. Screenshots are under
`docs/demo/screenshots/3c/`.

Raw-text tokenisation can differ from the dataset tokenisation. Organisation names,
unknown words and sentence splitting remain limitations. Current Indian-news
performance has not been independently verified. No IndicBERT work was started.
English and translation were subsequently authorized separately and are documented
in `MILESTONE_4A_LANGUAGE_TOOLS.md`; they do not alter these Hindi baselines.
