# Five-minute faculty demonstration

## 0:00–0:40 — Problem and task

“IndicNewsNER extracts complete person, organisation and location spans from Hindi
text. Hindi names lack capitalisation cues and often span several tokens.”
Point to the PER/ORG/LOC legend. State that this is the frozen classical baseline,
not the eventual multilingual or transformer system.

## 0:40–1:30 — Enter and process text

Choose **Person and place**. Read the sentence about मीरा शर्मा visiting a book
exhibition in जयपुर. Select **Extract Entities**. Explain that the original
Unicode text is tokenised, passed through frozen contextual CRF features, and
tagged with BIO labels. The example is original demo text, not a test-set example.

## 1:30–2:20 — Interpret output

Point to the blue person and green location highlights. Explain that both tokens
in मीरा शर्मा form one entity. Show total and unique counts. Mention that the model
can make mistakes and that examples are not guaranteed to generalise.

## 2:20–3:10 — Table and exports

Show the entity table and exclusive character end offsets. Explain the exact
substring check. Filter by LOC, then restore the filters. Download JSON or CSV.
Explain that JSON contains original text, token/BIO details, model identity,
timestamp and offset conventions. Downloads contain all entities.

## 3:10–3:50 — Additional cases

Choose **Multiple entities** and run it: Rahul Verma and Neha Gupta are PER;
Mumbai and Pune are LOC in the recorded output. Choose **Everyday sentence** to
show the recorded no-entity result. Use Clear to demonstrate input validation.

## 3:50–4:30 — Model information

Expand Model details. Explain the 100,000-record training sample, model checksum,
and separate offline results: validation micro F1 0.713770 and clean-test micro F1 0.745064. Strict correctness requires
both complete boundaries and the right type. Token accuracy is secondary.

## 4:30–5:00 — Limitations and roadmap

State explicitly that current-news performance has not been independently verified.
Explain tokenisation differences, organisation ambiguity, domain shift and
uncalibrated mean tag marginals. The model was frozen before final evaluation;
no test examples informed this demo. The next proposed stage is BiLSTM-CRF, then
IndicBERT, each requiring separate approval. No later model has started.
