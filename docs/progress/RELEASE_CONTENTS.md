# Release contents and exclusions

This initial public snapshot contains verified source, configuration, tests, the
Streamlit application, phase report, demo guide, five-minute script, six real
screenshots, aggregate results and reproducibility/freeze manifests. The exact
file inventory is RELEASE_FILES.txt in this directory.

Excluded from the release and its history:

- All raw, derived and quarantined dataset contents (empty placeholders only).
- Every native CRF model binary, virtual environment, cache and temporary file.
- Final test predictions and restricted example files.
- Generated validation error-analysis examples and token-text BIO repair reports.
- Local training/audit logs, secrets and credentials.

Excluded artifacts are preserved locally. Model restoration instructions are in
MILESTONE_2D_CLOSURE.md. The original baseline commit remains on the local
codex/crf-demo-2d branch; it is not an ancestor of this clean public snapshot.

The final report supersedes the earlier interim report, which remains as a dated
historical record. Earlier progress documents and test reports retain their
original counts and remote status; closure_verification.json and closure_pytest.xml
are the release checks. The package metadata is frozen at 0.1.0; the release tag
v0.2.0-crf-demo versions the added demonstration and safeguards.

The guard is mandatory for supported future development. It is not an OS sandbox;
stronger isolation requires mounting only the approved train/validation splits.
No new model training or final evaluation occurred during closure.
