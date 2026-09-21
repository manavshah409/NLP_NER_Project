# Milestone 2D — CRF demonstration completed

The frozen classical baseline now has a local Streamlit demonstration. No retraining, test-prediction access, tuning or later-model work occurred.

## Architecture

`app.py` coordinates the interface. `text_preprocessor.py` supplies Unicode tokens and sentence ranges with original-text offsets; `crf_predictor.py` verifies only the frozen runtime and performs locked tag/marginal inference; `entity_formatter.py` reconstructs strict spans and escapes offset-based HTML; `exporter.py` produces in-memory UTF-8 JSON/CSV.

Only `crf_100k_seed42_24dd05e84a26` is accepted. Frozen runtime/model checksums and existing package versions verify before inference; no dataset access is required. Confidence is the uncalibrated arithmetic mean of predicted-tag marginals.

## Actual original-example results

| Example | Original input | Actual predicted entities |
|---|---|---|
| Person and place | मीरा शर्मा ने जयपुर में पुस्तक प्रदर्शनी देखी। | मीरा शर्मा → PER; जयपुर → LOC |
| Organisation and place | भारतीय स्टेट बैंक ने भोपाल में नई शाखा खोली। | भारतीय स्टेट बैंक → ORG; भोपाल → LOC |
| Multiple entities | राहुल वर्मा और नेहा गुप्ता सोमवार को मुंबई से पुणे गए। | राहुल वर्मा → PER; नेहा गुप्ता → PER; मुंबई → LOC; पुणे → LOC |
| Punctuation and lines | ‘अनिता राव’, लखनऊ पहुंचीं। / उन्होंने कहा: बैठक कल होगी! | अनिता राव → PER; लखनऊ → LOC |
| Everyday sentence | आज मौसम सुहावना है और हल्की हवा चल रही है। | None detected |

## Validation

85 tests passed, 0 failed, 0 skipped, 0 errored. All original 66 tests remain intact. The first full demo run had 83 passes and two test-harness path failures; using an absolute path derived from the test file fixed them. The final full suite passed. `pip check` and compile checks passed.

Browser verification covered page loading, example selection, extraction, empty-input errors, highlights, summary, table, JSON and CSV download events, and the offline metrics panel. AppTest additionally checked Clear, input length rejection and no-entity output. Screenshots in docs/demo/screenshots are real unedited captures; the browser viewport changed during the session.

## Commands

```sh
.venv/bin/python -m pip install --constraint requirements.txt streamlit==1.63.0
.venv/bin/python -m pytest --junitxml=reports/demo/pytest.xml
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q app.py src/inference tests/test_demo.py tests/test_demo_app.py
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

The initial sandbox launch could not bind the local port; the permitted loopback launch succeeded. After activating the virtual environment the normal command is `streamlit run app.py`.

## Files and limitations

Added app.py, four inference modules, requirements-demo.txt, .streamlit/config.toml, two test files, demo reports, five screenshots, the demo guide and faculty script. README and CHANGELOG were updated. All additions sit outside the frozen source file list.

The tokenizer is Unicode-aware but cannot reconstruct arbitrary benchmark tokenisation exactly. Periods in abbreviations/decimals can create extra sentence breaks. Limits are 5,000 Unicode code points and 300 tokens per sentence. Offsets are code points, not bytes or UTF-16 indices. Marginals are not calibrated span correctness. Unique entities use exact (text,type) pairs. CSV formula protection adds an apostrophe where needed; JSON preserves exact text. Current-news and English performance remain unverified.

No remote was configured at verification; push requires a GitHub repository URL. The local server and demo browser tab were left available for faculty rehearsal. Stop here; a later model requires separate approval.
