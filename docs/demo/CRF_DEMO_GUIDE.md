# IndicNewsNER CRF demonstration guide

## Install and launch

Run from the repository root on the verified Python 3.14 macOS arm64 environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python -m pip check
streamlit run app.py
```

Open http://127.0.0.1:8501. The committed Streamlit configuration binds only to
loopback and disables usage statistics. The browser does not need internet for
inference. The app was launched and checked locally with the equivalent command:

```sh
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

`requirements.txt` remains frozen. `requirements-demo.txt` adds Streamlit 1.63.0
and its resolved dependencies while retaining every original package version.

## Model placement and verification

The only accepted experiment is `crf_100k_seed42_24dd05e84a26`. Its existing
`model.crfsuite`, `resolved_config.yaml` and `environment.json` must be present in:

```text
models/crf/crf_100k_seed42_24dd05e84a26/
```

The model binary is 22,707,636 bytes and is deliberately excluded from Git. A fresh
clone needs the original binary from the project owner, not a substitute model.
The committed freeze manifest and original runtime source must also be present.
The app validates the freeze envelope, exact model hash, configuration, runtime
source and package versions before loading and on inference. A mismatch disables
inference; there is no fallback. It never loads training or test datasets.

## Faculty workflow

Choose an original example or type Hindi text, then select **Extract Entities**.
The page shows original text with blue PER, orange ORG and green LOC highlights;
counts for all mentions and unique (text, type) pairs; a filterable entity table;
and JSON/CSV downloads. The model-details and limitations panels are below.
**Clear** removes the current input and output. Editing text makes the old result
stale until extraction is requested again. Downloads always contain all entities,
not just the filtered table rows.

Example input:

```text
मीरा शर्मा ने जयपुर में पुस्तक प्रदर्शनी देखी।
```

The actual frozen-model output is मीरा शर्मा → PER and जयपुर → LOC. The five
built-in examples were authored for this demo; they were not copied from test
predictions and are not claims about real events. Full actual outputs are saved
in `reports/demo/original_example_predictions.json`.

## Offsets, tokenisation and confidence

All character offsets count Python Unicode code points, start at zero and use
exclusive ends. Token indices also use exclusive ends and are global across
sentences. Each entity is exactly `original_text[start_char:end_char]`, preserving
spaces inside multi-token spans. These indices are not UTF-8 byte offsets or
JavaScript UTF-16 indices; integrations must convert them if needed.

The tokenizer groups Unicode letters, combining marks and numbers, keeps joiners
inside word runs, skips whitespace while retaining its offsets, and makes
punctuation/symbols separate tokens. Terminal punctuation and line breaks delimit
sentences. No Unicode normalisation, lowercasing, transliteration or whitespace
replacement is applied to the original input. The frozen feature extractor may
produce its original Latin lowercase feature without modifying input.

The maximum input is 5,000 code points, with at most 300 tokens per sentence.
These are conservative interactive limits, not trained-model limits. Abbreviations
and decimals containing periods can be split into separate sequences. The raw-text
tokenizer cannot exactly reproduce the benchmark's supplied tokenisation; this is
a deployment limitation. Invalid predicted I-tags never silently start new spans.

Entity confidence is the arithmetic mean of the CRFsuite marginal probabilities
for the predicted tags of its constituent tokens, obtained immediately after each
sentence is tagged. This is not a joint span probability, calibrated accuracy or
factual truth. A lock keeps cached tagger/marginal operations together across
sessions. JSON includes the BIO sequence, token marginals and definition.

## Privacy, exports and limitations

Input and result data are kept in memory only; the app does not write them to disk
or send them to an external inference service. Users explicitly download exports.
HTML is escaped before offset-based highlighting. CSV uses UTF-8 with a BOM for
spreadsheet compatibility; formula-like entity text gets a leading apostrophe.
JSON retains exact text. Unique counts distinguish entity type and exact spelling.

Offline validation micro F1 is 0.713769728 and clean-test micro F1 is 0.745063624.
These are benchmark results, not performance on the current input. Naamapadam is
multi-domain and current Indian-news performance is unverified. ORG and boundary
ambiguity remain limitations. English and neural models are not implemented here.

## Troubleshooting

- Model verification error: restore the original model/configuration/source and
  pinned package versions. Never disable the checksum check.
- Missing model after clone: obtain the frozen binary from the project owner.
- Port occupied: stop the earlier local app or use `--server.port 8502`.
- Empty or over-limit input: follow the visible validation message.
- No entities: a legitimate possible prediction, not proof that no entity exists.
- Browser shows old output: select Extract Entities again; refresh after code changes.

Do not run final test inference for a demo. Test predictions and illustrations
remain restricted by AGENTS.md.
