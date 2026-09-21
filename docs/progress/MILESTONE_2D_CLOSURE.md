# Milestone 2D closure

The deliverable is the frozen Hindi CRF demonstration, not a new model. The report
in docs/reports covers data preparation, features, scaling, separate validation
and final metrics, interface, verification and limitations. The five-minute
script is docs/demo/FACULTY_DEMONSTRATION_SCRIPT.md.

## Sealed evaluation policy

All future development commands must install src.development_guard.install()
before opening data. The supported wrapper is:

    .venv/bin/python -m scripts.development --module src.training.train_crf --help

The wrapper does not authorise training. The frozen historical training command
has whole-dataset checksum preflight and therefore intentionally fails under the
guard if actually run. It is retained only for historical reproducibility. Direct
legacy commands are retired for development. A separately approved future model
must use the guard and training/validation-only preflight, with subprocesses
launched independently under the same policy or an isolated container.

The audit hook denies raw data, all derived files except canonical train_clean
and validation_clean, and the entire final evaluation directory (including
metrics). It catches builtins.open, pathlib and os.open, resolved symlinks,
and dataset writes, and denies child-process escape. Final aggregate reports
remain available for reporting outside the guarded development process.

This is an accidental-access safeguard, not an OS security sandbox. It cannot
prevent a person bypassing the wrapper, copying test contents to a new filename,
or using native code to read files. For strong isolation, mount only train_clean
and validation_clean in a separate development container. Never mount final
artifacts, raw datasets or final aggregate metrics there. No retrospective
renaming or permission changes to frozen files are performed.

New final evaluations require explicit approval and must retain the one-shot
controls. Do not reopen predictions or restricted examples during later model
selection. No BiLSTM-CRF, IndicBERT or English work is part of this release.

## Model restoration

The model is excluded from Git and release assets pending resolution of upstream
license discrepancies. An authorised existing holder can copy the exact original
model.crfsuite into models/crf/crf_100k_seed42_24dd05e84a26/ and verify SHA256
80395321d26b43076c01cdbea5d3d729159ef3581ca9fdec1214d3cec92cac5c.
The app verifies it automatically and refuses fallback. A fresh clone alone
cannot run inference without this artifact. Retraining is not restoration of the
frozen artifact and is not authorised here. Full freeze verification additionally
requires the original local datasets; demo runtime checks do not require them.

## Verification and release preparation

Final closure suite: 110 passed, 0 failed, 0 skipped, 0 errors. All 51 frozen
files passed SHA256 verification and pip check found no broken requirements.
An initial sandbox run had 107 passes and one process-monitor permission failure;
a permitted rerun passed, followed by the expanded 110-test final run.

The remote was inspected and fetched while empty. The intended initial default
branch is main. A new root release snapshot preserves the historical local branch
while excluding benchmark-text error reports and BIO repair text from published
history. No local benchmark artifact was deleted or edited. pyproject.toml remains
at frozen version 0.1.0; the annotated Git tag versions the demo separately.

JSON and CSV download events were verified with the original person/place demo.
Six unedited screenshots cover interface, input, highlights, table, model details
and export controls. The final report embeds the required interface evidence.
