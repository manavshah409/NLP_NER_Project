# Implementation log

## Workspace inspection and environment

The workspace initially contained only an uncommitted Git repository, with no
tracked or untracked project files. No earlier implementation or tests existed.
Python 3.14.7, macOS 26.6.2 arm64, 24 GiB physical RAM and approximately 836 GB
free disk were observed. Package versions and available memory are recorded in
`environment/environment.json`. No CUDA or neural framework was installed.

Commands used for initial inspection:

```sh
ls -la
git status --short
git log -1 --oneline
rg --files --hidden -g '!.git' -g '!node_modules' -g '!.venv'
command -v python3.12 python3.13 uv
ls /Library/Frameworks/Python.framework/Versions
sysctl -n hw.memsize
vm_stat
```

Python's `sys`, `platform`, `shutil.disk_usage` and `importlib.metadata` were also
used to inspect the interpreter, architecture, disk and installed dependencies.
`git log` confirmed there were no commits. `sysctl` required sandbox permission.

## Initialisation and dependency resolution

Created the modular structure directly in this repository root; no redundant
nested repository was created. Installed current compatible wheels into `.venv`,
then pinned the resolved package set. PyTorch and transformer dependencies are
deferred until those milestones are approved. The local strict evaluator replaces
a seqeval dependency and is tested with hand-computed expected scores.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install datasets numpy pandas pyyaml scikit-learn sklearn-crfsuite python-crfsuite psutil pytest
.venv/bin/python -m pip freeze > requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m src.utils.environment
```

Network DNS was blocked in the sandbox; installation succeeded with permission.
`pip check` reported no broken requirements. The pip cache warning reflects the
sandbox cache directory and does not affect the installed environment.

## Acquisition, audit, normalisation and construction

```sh
.venv/bin/python -m src.data.acquire
.venv/bin/python -m src.data.audit > reports/data_audit/audit.log 2>&1
.venv/bin/python -m src.data.derive > reports/derived_data/derive.log 2>&1
.venv/bin/python -m scripts.verify_reproducibility > reports/derived_data/reproducibility.log 2>&1
.venv/bin/python -m src.data.derive > reports/derived_data/reverification.log 2>&1
```

Acquisition needed network permission. The Python installation's missing CA trust
configuration initially caused a certificate error. Using certifi's CA bundle
fixed the error without disabling TLS verification. The official archive and its
source JSONL files were preserved; raw labels are strings in `words`/`ner` fields.
The official loader dynamically supplies the label ordering via syntax inspection.

The audit calculated raw counts and errors. Unit tests verified repair and removal
policies before derived construction. Construction verified raw integrity before
and after, source-index accounting, exact source contents and clean separation.
Independent streaming reconstruction checked the clean output hashes again.
The checkpoint JSON and Markdown contain exact counts and reference comparisons.

## CRF implementation and verification

```sh
.venv/bin/python -m pytest --junitxml=reports/environment/pytest.xml
.venv/bin/python -m scripts.checkpoint_report
.venv/bin/python -m src.training.train_crf --size 1000
```

The native CRFsuite worker is supervised by a separate process to avoid relying on
a Python thread while native training holds the GIL. Resource thresholds use live
available memory. Test and training process inspection requires permission outside
the sandbox. A first sandbox attempt was terminated before fitting and preserved
in a directory ending `_interrupted_sandbox`.

A 1k preflight then caught integer-versus-string histogram keys after JSON loading.
The statistics representation was fixed and a regression test added. This changed
neither data contents nor counts. The failed attempt was preserved without a
completion marker. Only directories with `complete.json` represent successful runs.

Exact current test counts are saved in JUnit XML and the checkpoint. No previous
54-test result was assumed. Synthetic fixtures test code, not model performance.

## Controlled scaling

```sh
.venv/bin/python -m src.training.train_crf --size 10000
.venv/bin/python -m src.training.train_crf --size 50000
.venv/bin/python -m scripts.audit_comparison
.venv/bin/python -m compileall -q src scripts tests
```

The 1k run completed with strict validation micro F1 0.5188805865502114,
3.825170416996116 seconds fitting time and 116752384 bytes sampled peak RSS.
The 10k run completed with strict validation micro F1 0.6389986582608856,
39.68323516700184 seconds fitting time and approximately 0.419 GiB sampled peak RSS.
The 50k run was then launched with identical source, features and model parameters.
Its results will be taken only from its completion report.

`reports/crf/source_snapshot.json` preserves the shared source code for successful
controlled runs because the initially empty Git repository has no prior commit.
The before/after supplemental audit and compile check completed successfully.
Markdown was searched for escaped underscores, duplicated SVG markers and malformed URL suffixes;
no such rendering artifacts were found in generated project Markdown.

## Completion and final verification

The 50k experiment completed: strict validation micro F1 0.6951455614513122,
approximately 199.76 seconds fitting time and 1.488 GiB sampled peak RSS.
All three experiments used the same feature/model settings and nested samples.
The final comparison recommends a 100k run subject to approval; it has not started.

The final scripts were executed sequentially with this command:

```sh
.venv/bin/python -c 'from scripts.compare_crf import main as compare; from scripts.verify_experiments import main as verify; from scripts.completion_report import main as report; compare(); verify(); report()'
```

The scripts generated the scaling CSV/Markdown, verified artifact checksums,
reproduced all three experiment identifiers from the retained source and dependency
snapshot, confirmed native save/reload consistency on training-only inputs, and
updated the project model card and completion report. All checks passed.

The full suite collected 53 tests: 53 passed, 0 failed, 0 skipped and 0 errored.
The exact JUnit results are preserved. No tests were added simply to reproduce the
reference count. The final status is in `milestone_2b_completion.md` and its JSON.
No 100k, 250k, full-data, test-prediction, neural-model or English run was performed.

## Approved 100k extension

The user's subsequent “continue” approved the specifically recommended 100k
experiment. The size guard was extended to 100k, and per-run source snapshots were
added so the new experiment ID remains reproducible without changing old snapshots.
An AST comparison confirms the worker function is unchanged; all other source
modules are byte-identical to the original controlled experiments. The earlier
comparison is preserved under `crf/checkpoint_050k/`.

```sh
.venv/bin/python -m compileall -q src scripts
.venv/bin/python -m pytest --junitxml=reports/environment/pytest.xml
.venv/bin/python -m src.training.train_crf --size 100000
```

Compilation succeeded and all 53 tests passed. The 100k experiment was then
launched with unchanged features and model parameters, a nested seed-42 sample,
live memory monitoring and validation-only evaluation. Larger sizes and all test
evaluation remain unapproved.

The 100k run completed successfully as `crf_100k_seed42_24dd05e84a26`.
Observed strict validation micro F1: 0.7137697275946124; macro F1: approximately
0.709317. Fitting took 368.14 seconds and sampled peak RSS was 2.658 GiB.
The micro F1 gain over 50k was 0.018624 (1.8624 percentage points).

The comparison, verification and completion scripts were executed again, followed
by raw/derived checksum verification. All four native models reload consistently,
their experiment IDs reproduce from the saved source snapshots, their artifact
checksums match, and the data checksums remain unchanged. Reports and model card
now include the approved extension. The recommendation is to stop CRF scaling at
100k for now under the documented conservative resource rule. No larger run,
test prediction or later-model work has started.
