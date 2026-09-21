# IndicNewsNER

Fourth-year undergraduate project: named entity recognition for Indian news.
The current implementation focuses on Hindi and the PER, ORG and LOC types.
English, neural models and the demonstration interface are future milestones.
Naamapadam contains multiple domains and is not exclusively an Indian-news dataset.

The data checkpoint and authorised CRF scaling milestone are complete. Strict
validation micro F1 is 0.518881 at 1k, 0.638999 at 10k, 0.695146 at 50k and
0.713770 at the approved 100k scale.
All 66 tests pass. See [completion report](reports/milestone_2b_completion.md),
[scaling comparison](reports/crf/crf_scaling_comparison.md) and
[data checkpoint](reports/checkpoint_milestones_1_2a.md). The current recommendation is to stop CRF scaling at 100k. Both test splits were subsequently evaluated once under the approved Milestone 2C freeze.

## Setup

Commands below run from the repository root. The verified development environment
uses Python 3.14 on macOS arm64. `requirements.txt` pins the installed dependency
set, including transitive dependencies. No CUDA dependency is required for CRF.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m src.utils.environment
.venv/bin/python -m pytest
```

PyTorch, Transformers, Accelerate and Evaluate are deferred until their approved
milestones; they are not needed for classical CRF. Their compatible versions and
MPS support must be checked then. Strict evaluation is implemented locally and
tested directly, so seqeval is not required.

## Reproduce data preparation

```sh
.venv/bin/python -m src.data.acquire
.venv/bin/python -m src.data.audit
.venv/bin/python -m src.data.derive
.venv/bin/python -m pytest --junitxml=reports/environment/pytest.xml
```

Acquisition resolves and records the official repository revision, downloads the
Hindi archive, retains its original JSONL files, loader source and dataset card,
and discovers labels from the loader's syntax without executing it. The stored
provenance records the resolved revision. Set `revision` in the YAML to that hash
to reproduce acquisition from the same upstream source. Existing raw data are
verified rather than overwritten. An interrupted download stays in an explicitly
named `.incomplete` directory; it is never treated as complete data.

Raw checksums are SHA-256 per file, keyed by relative POSIX path. The combined
hash is SHA-256 of UTF-8 JSON of that mapping, sorted keys, no whitespace, and
`ensure_ascii=False`. The manifest lives outside the raw directory. This is not
a filesystem directory hash. Provenance timestamps mean independently acquired
raw directories can have different combined hashes even when data files match.
Compare individual source-file hashes for cross-acquisition identity.

Strict raw BIO only starts entities at B tags. Derived CoNLL normalisation repairs
invalid I tags to B tags, with each change logged before any records are removed.
Text overlap uses NFC normalisation, joins tokens with spaces and collapses
whitespace, preserving case and original stored tokens.

Derived order: normalise; quarantine empty test records; remove overlap with
earlier clean splits (train before validation before test); remove exact token
and label duplicates, retaining the lowest source index. The complete official
test is separately preserved without repair. All rows retain original indices.
Test data are used only for integrity preparation, never model development.

Data and model binaries are excluded from Git. Source, configs, tests and reports
are separated. Synthetic test fixtures are not benchmark observations.

## Evaluation and scope

The controlled CRF commands are:

```sh
.venv/bin/python -m scripts.verify_reproducibility
.venv/bin/python -m src.training.train_crf --size 1000
.venv/bin/python -m src.training.train_crf --size 10000
.venv/bin/python -m src.training.train_crf --size 50000
.venv/bin/python -m src.training.train_crf --size 100000
.venv/bin/python -m scripts.compare_crf
```

Run them sequentially and inspect each completion before scaling. The worker is
monitored from a separate process; macOS sandbox permissions must allow process
inspection. The conservative memory budget is the smaller of half physical RAM
and 60% of currently available RAM. Stop occurs at 90% of that budget or when
available RAM falls below 10% of physical RAM. The exact threshold is saved per
run. RSS peaks are sampled, not exact operating-system high-water marks.

Samples reserve seed-hash-selected coverage for PER, ORG and LOC, then fill by
seed-hash order. They are nested and retain original source indices. Full and
sampled label/span distributions are saved in each sample manifest. This is
coverage-aware sampling, not class-balanced resampling.

Training is streamed into the native CRFsuite trainer to avoid keeping Python
feature dictionaries for the whole sample. Saved models use `.crfsuite`, with
configuration, environment and metrics alongside them. An experiment ID hashes
the source, parameters, sample membership, derived checksum and dependency lock.
Completed runs have a checksummed `complete.json` report; interrupted runs remain
preserved and are never silently overwritten.

Strict entity micro F1 is primary: both full boundary and type must match.
Macro F1 includes all three entity types, with zero for unsupported types.
Token accuracy is secondary. Only clean validation is permitted for development.
The 100k extension was approved after the initial comparison. Any 250k or
full-data training, either test evaluation, later models, English and the final
interface still require separate approval.

## Sources and licensing

- [Official Naamapadam dataset](https://huggingface.co/datasets/ai4bharat/naamapadam)
- [Naamapadam paper](https://aclanthology.org/2023.acl-long.582/)
- [CRFsuite Python binding](https://github.com/scrapinghub/python-crfsuite)

The Hub metadata advertises CC0-1.0, while the official loader declares
Creative Commons Attribution-NonCommercial 4.0. Retained upstream originals
document the conflict; no blanket redistribution permission is asserted here.

See `reports/implementation_log.md` and generated data reports for observed status.

## Frozen final test evaluation — Milestone 2C

The 100k baseline is frozen. Each test split was evaluated once, without training or tuning.

| Split | Strict micro F1 | Strict macro F1 |
|---|---:|---:|
| test_clean | 0.745063624 | 0.732890036 |
| official_test | 0.766889186 | 0.759028464 |

The clean result is the main Naamapadam conclusion. The official result is a separate overlapping/raw-BIO benchmark comparison. All 66 tests pass. See [final report](reports/crf/CRF_BASELINE_FINAL_REPORT.md). Full-precision metrics are in the JSON reports.

## Local CRF demonstration — Milestone 2D

The presentable Streamlit demo is complete. It uses only the frozen 100k model,
preserves Unicode text and offsets, highlights PER/ORG/LOC, and exports JSON/CSV.
All 85 tests pass. The original baseline files and dependency pins are unchanged.

```sh
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
streamlit run app.py
```

Open http://127.0.0.1:8501. See [demo guide](docs/demo/CRF_DEMO_GUIDE.md),
[five-minute faculty script](docs/demo/FACULTY_DEMONSTRATION_SCRIPT.md) and
[completion report](docs/progress/MILESTONE_2D_CRF_DEMO.md).
The original binary must be present locally; there is no fallback model. No user
input is saved to disk by default. Test predictions are never opened by the demo.

## Milestone 2D release

The phase report is [IndicNewsNER project report](docs/reports/IndicNewsNER_Milestone_2D_Final_Report.docx).
See [closure and sealed evaluation policy](docs/progress/MILESTONE_2D_CLOSURE.md)
for supported guarded development commands, their limits and model restoration.
The model binary and all datasets are excluded from this repository. A fresh
clone requires the exact original model from an authorised holder before running
the demo. Current-news performance has not been independently verified.
