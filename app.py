"""Local faculty demonstration of the frozen Hindi CRF baseline."""
import streamlit as st
from src.inference.crf_predictor import CRFPredictor, EXAMPLES, EXPERIMENT
from src.inference.entity_formatter import highlight
from src.inference.exporter import json_export, csv_export
from src.inference.text_preprocessor import MAX_CHARACTERS

st.set_page_config(page_title="IndicNewsNER · Hindi NER", page_icon="📰", layout="wide")
st.markdown("""<style>
.block-container{max-width:1140px;padding-top:2.4rem;padding-bottom:3rem}
h1{letter-spacing:-.035em;font-size:2.4rem!important}
[data-testid="stMetricValue"]{font-size:1.7rem}
.eyebrow{color:#64748b;font-size:.78rem;font-weight:650;letter-spacing:.13em;text-transform:uppercase}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin:12px 0 18px;font-size:.9rem}
.legend span{padding:4px 10px;border-radius:5px}
textarea{line-height:1.8!important;font-size:1.05rem!important}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">IndicNewsNER / Research demonstration</div>', unsafe_allow_html=True)
st.title("IndicNewsNER — Hindi Named Entity Recognition")
st.markdown("**Classical CRF Baseline Demonstration**")
st.caption("Explore people, organisations and places in Hindi text. Runs locally with the frozen 100k baseline.")


@st.cache_resource
def get_predictor():
    return CRFPredictor()


try:
    predictor = get_predictor()
    from src.inference.crf_predictor import verify_runtime
    verify_runtime()
except Exception as error:
    st.error(f"Model verification failed. Inference is disabled. {error}")
    st.stop()


def change_example():
    st.session_state["news_text"] = EXAMPLES.get(st.session_state["example"], "")
    st.session_state.pop("result", None)


def clear():
    st.session_state["news_text"] = ""
    st.session_state["example"] = "Write your own"
    st.session_state.pop("result", None)


with st.container(border=True):
    st.subheader("1 · Enter Hindi text")
    st.selectbox("Try an original example", ["Write your own"] + list(EXAMPLES), key="example", on_change=change_example)
    text = st.text_area("Hindi news text", key="news_text", height=155, placeholder="यहाँ हिंदी समाचार का पाठ लिखें…")
    st.caption(f"{len(text):,} / {MAX_CHARACTERS:,} characters · Original spacing and line breaks are preserved. Text stays in this session; it is not saved to disk.")
    left, right, _ = st.columns([2, 1, 5])
    extract = left.button("Extract Entities", type="primary", width="stretch")
    right.button("Clear", on_click=clear, width="stretch")
    if extract:
        st.session_state.pop("result", None)
        try:
            with st.spinner("Extracting entities…"):
                st.session_state["result"] = predictor.predict(text)
        except ValueError as error:
            st.error(str(error))
        except Exception:
            st.error("Inference could not complete. Check the model artifacts and restart the app.")

st.markdown('<div class="legend"><span style="background:#dbeafe;color:#1e40af">PER · Person</span><span style="background:#ffedd5;color:#9a3412">ORG · Organisation</span><span style="background:#dcfce7;color:#166534">LOC · Location</span></div>', unsafe_allow_html=True)
result = st.session_state.get("result")
if result and result["text"] != text:
    st.info("The input has changed. Select Extract Entities to refresh the results.")
elif result:
    with st.container(border=True):
        st.subheader("2 · Entities in context")
        st.markdown(highlight(result["text"], result["entities"]), unsafe_allow_html=True)
    entities = result["entities"]
    columns = st.columns(5)
    summary = [("Total entities", len(entities))] + [(k, sum(e["type"] == k for e in entities)) for k in ("PER", "ORG", "LOC")]
    summary.append(("Unique entities", len({(e["text"], e["type"]) for e in entities})))
    for column, (label, value) in zip(columns, summary):
        column.metric(label, value)
    st.caption("Unique entities count exact (text, type) pairs; repeated mentions remain separate rows.")
    st.subheader("3 · Review and export")
    types = st.multiselect("Filter entity types", ["PER", "ORG", "LOC"], default=["PER", "ORG", "LOC"])
    rows = [{"Entity": e["text"], "Type": e["type"], "Start": e["start_char"], "End": e["end_char"], "Mean tag marginal": round(e["confidence"], 4)} for e in entities if e["type"] in types]
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    elif not entities:
        st.info("No entities detected. This is a model prediction, not proof that the text contains none.")
    else:
        st.info("No entities match the selected filters.")
    st.caption("Offsets are zero-based Unicode code points; ends are exclusive. Mean tag marginals express model certainty, not calibrated entity accuracy or factual truth.")
    one, two, _ = st.columns([1, 1, 3])
    one.download_button("Download JSON", json_export(result), "indicnewsner_result.json", "application/json", width="stretch")
    two.download_button("Download CSV", csv_export(result), "indicnewsner_entities.csv", "text/csv", width="stretch")
    st.caption("Downloads include all detected entities, regardless of the table filter.")
else:
    st.info("Choose an example or enter Hindi text, then select Extract Entities.")

with st.expander("Model details · offline evaluation", expanded=False):
    st.markdown(f"**Classical CRF · Hindi · PER / ORG / LOC**\n\nExperiment: `{EXPERIMENT}`\n\nTraining sample: **100,000** clean Naamapadam records. Model size: **22,707,636 bytes (21.66 MiB)**.")
    a, b = st.columns(2)
    a.metric("Validation strict micro F1", "0.713770")
    b.metric("Clean test strict micro F1", "0.745064")
    st.caption("These are frozen offline benchmark results, not accuracy measurements for your current input. Runtime checksums verified; no fallback model.")
with st.expander("Limitations and interpretation", expanded=False):
    st.markdown("Naamapadam is multi-domain. Performance on current Indian news has not been independently verified. Organisation names and entity boundaries can be ambiguous.\n\nRaw text uses Unicode-aware tokenisation; it may differ from the benchmark's supplied tokens. Sentence breaks use terminal punctuation and newlines, so abbreviations can create extra boundaries. Hindi is supported; English capability is not claimed.\n\nInvalid predicted I-tags do not silently start new entities. Text is kept only in session memory. CSV formula-like text is prefixed with an apostrophe for spreadsheet safety; JSON preserves exact text.")
