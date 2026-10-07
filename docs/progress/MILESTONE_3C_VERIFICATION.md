# Final verification — 2026-10-07

- Complete suite: **170 passed, 0 failed, 0 skipped**, 13.09 seconds.
  Machine-readable report: `reports/demo_3c_tests.xml`.
- Both environments: `pip check` reports no broken requirements.
- Frozen manifests: pinned envelopes pass; all 38 classical and 55 neural
  non-dataset files match. Sealed dataset files deliberately not opened/rehashed.
- Hindi CPU inference and live MPS inference verified. Authored original input
  produced matching person/place spans in the live comparison.
- Live comparison JSON downloaded and parsed; both models and exact offsets checked.
- Live English CSV downloaded and parsed: three rows with PER/ORG/LOC.
- Live translation output and warning observed. Name/place corruption is documented
  as a quality failure. Smoke success is not evidence of translation correctness.
- Regression tests cover English CSV schema, stale UI output, missing environment,
  failed checksums, timeouts, offline subprocess flags, limits and exact offsets.
- Local translation restoration script verifies all downloaded files against the
  committed manifest. Model directory and isolated environment are Git-ignored.
- `git diff --check` passes. No frozen source/config/weights/environment changes.
- Branch: `main`. Remote: `https://github.com/manavshah409/NLP_NER_Project.git`.
  Publication checkpoint: reviewed for a normal push to main on 2026-10-07.
  The resulting commit is recorded in Git history; no new release tag is requested.

## Files created or modified in this work package

- `.gitignore`
- `CHANGELOG.md`
- `README.md`
- `app.py`
- `configs/language_tools_manifest.json`
- `docs/MODEL_CARD_DEMO_ADAPTERS.md`
- `docs/demo/DUAL_MODEL_DEMO_GUIDE.md`
- `docs/demo/LANGUAGE_TOOLS.md`
- `docs/demo/screenshots/3c/english-ner.jpg`
- `docs/demo/screenshots/3c/hindi-comparison.jpg`
- `docs/demo/screenshots/3c/hindi-to-english.jpg`
- `docs/progress/MILESTONE_3C_DUAL_MODEL_DEMO.md`
- `docs/progress/MILESTONE_3C_VERIFICATION.md`
- `docs/progress/MILESTONE_4A_LANGUAGE_TOOLS.md`
- `pages/1_English_and_Translation.py`
- `reports/demo_3c_tests.xml`
- `requirements-language.lock.txt`
- `scripts/setup_language_models.py`
- `src/inference/bilstm_crf_predictor.py`
- `src/inference/comparison.py`
- `src/inference/language_bridge.py`
- `src/inference/runtime_verification.py`
- `src/language_tools/__init__.py`
- `src/language_tools/worker.py`
- `tests/conftest.py`
- `tests/test_bilstm_crf.py`
- `tests/test_dual_demo.py`
- `tests/test_language_tools.py`

## Exclusions

`.venv-language/` and all `models/language_tools/` contents are excluded, including
translation weights, tokenizer files and local metadata. Existing dataset, secret,
cache and restricted-report exclusions remain in force. Hindi model binaries stay
excluded. Download verification used authored examples only; user-entered text is
not committed. No project test evaluation or training was run.

## Evidence

Three actual browser captures are in `docs/demo/screenshots/3c/`. Hindi comparison
shows both model outputs, English shows detected entities, and translation shows
its warning and observed imperfect output. These images are not mockups.

## Recommended next work

Human-review a small independently authored development set for English NER and
Hindi-to-English translation, particularly name/place preservation; evaluate a
stronger translator before calling this feature reliable. Keep those development
artifacts separate from all sealed Hindi final-evaluation data. New final testing
requires explicit authorization. IndicBERT remains unstarted.
