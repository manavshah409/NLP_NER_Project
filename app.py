"""Local comparison of two frozen Hindi NER baselines; no dataset access."""
import streamlit as st
from src.inference.crf_predictor import EXAMPLES
from src.inference.comparison import ClassicalPredictor, compare_entities
from src.inference.entity_formatter import highlight
from src.inference.exporter import json_export, csv_export
from src.inference.text_preprocessor import MAX_CHARACTERS, tokenize, sentence_ranges
from src.inference.runtime_verification import CRF_ID, NEURAL_ID

st.set_page_config(page_title='IndicNewsNER · Hindi NER', page_icon='📰', layout='wide')
st.markdown('''<style>
.block-container{max-width:1400px;padding-top:2.4rem;padding-bottom:3rem}
h1{letter-spacing:-.035em;font-size:2.4rem!important}
[data-testid="stMetricValue"]{font-size:1.6rem}
.eyebrow{color:#64748b;font-size:.78rem;font-weight:650;letter-spacing:.13em;text-transform:uppercase}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin:12px 0 18px;font-size:.9rem}
.legend span{padding:4px 10px;border-radius:5px}
textarea{line-height:1.8!important;font-size:1.05rem!important}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">IndicNewsNER / Research demonstration</div>', unsafe_allow_html=True)
st.title('IndicNewsNER — Hindi Named Entity Recognition')
st.markdown('**Two frozen baselines · One Hindi input**')
st.caption('Explore people, organisations and places with Classical CRF, BiLSTM-CRF or a side-by-side comparison.')


@st.cache_resource
def get_predictor(family):
    if family == 'Classical CRF':
        return ClassicalPredictor()
    from src.inference.bilstm_crf_predictor import BiLSTMCRFPredictor
    return BiLSTMCRFPredictor()


def reset_results():
    st.session_state.pop('results', None)


def change_example():
    st.session_state['news_text'] = EXAMPLES.get(st.session_state['example'], '')
    reset_results()


def clear():
    st.session_state['news_text'] = ''
    st.session_state['example'] = 'Write your own'
    reset_results()


mode = st.radio('Select model', ['Classical CRF', 'BiLSTM-CRF', 'Compare Both'], horizontal=True,
                key='model_mode', on_change=reset_results)
families = ['Classical CRF', 'BiLSTM-CRF'] if mode == 'Compare Both' else [mode]
predictors = {}
for family in families:
    try:
        predictor = get_predictor(family)
        predictor.verify()  # Recheck cached artifacts; never silently use a changed checkpoint.
        predictors[family] = predictor
    except Exception as error:
        st.error(f'{family} unavailable. Restore its original frozen model/configuration and restart the app. {error}')
if len(predictors) < len(families) and predictors:
    st.warning('Only the available baseline can run. Comparison requires both verified models; no model is substituted.')

with st.container(border=True):
    st.subheader('1 · Enter Hindi text')
    st.selectbox('Try an original example', ['Write your own'] + list(EXAMPLES), key='example', on_change=change_example)
    text = st.text_area('Hindi news text', key='news_text', height=155, placeholder='यहाँ हिंदी समाचार का पाठ लिखें…')
    st.caption(f'{len(text):,} / {MAX_CHARACTERS:,} characters · Original spacing and line breaks are preserved. Text stays in this session; it is not saved to disk.')
    left, right, _ = st.columns([2, 1, 5])
    extract = left.button('Extract Entities', type='primary', width='stretch', disabled=not predictors)
    right.button('Clear', on_click=clear, width='stretch')
    if extract:
        reset_results()
        try:
            list(sentence_ranges(text, tokenize(text)))
            results = {}
            with st.spinner('Extracting entities…'):
                for family, predictor in predictors.items():
                    try:
                        results[family] = predictor.predict(text)
                    except Exception as error:
                        st.error(f'{family} inference stopped: {error}')
            st.session_state['results'] = results
        except ValueError as error:
            st.error(str(error))

st.markdown('<div class="legend"><span style="background:#dbeafe;color:#1e40af">PER · Person</span><span style="background:#ffedd5;color:#9a3412">ORG · Organisation</span><span style="background:#dcfce7;color:#166534">LOC · Location</span></div>', unsafe_allow_html=True)


def show_result(result):
    family = result['model_family']
    st.subheader(family)
    st.caption(f"{result['experiment_id']} · {result['diagnostics']['device'].upper()}")
    with st.container(border=True):
        st.markdown('#### 2 · Entities in context')
        st.markdown(highlight(result['text'], result['entities']), unsafe_allow_html=True)
    entities = result['entities']
    summary = [('Total entities', len(entities))] + [(k, sum(e['type'] == k for e in entities)) for k in ('PER', 'ORG', 'LOC')]
    summary.append(('Unique entities', len({(e['text'], e['type']) for e in entities})))
    for column, (label, value) in zip(st.columns(5), summary):
        column.metric(label, value)
    st.caption('Unique entities count exact (text, type) pairs; repeated mentions remain separate rows.')
    st.markdown('#### 3 · Review and export')
    types = st.multiselect('Filter entity types', ['PER', 'ORG', 'LOC'], default=['PER', 'ORG', 'LOC'], key=f'filter_{family}')
    rows = []
    for entity in entities:
        if entity['type'] not in types:
            continue
        row = {'Entity': entity['text'], 'Type': entity['type'], 'Start': entity['start_char'], 'End': entity['end_char']}
        if family == 'Classical CRF':
            row['Mean tag marginal'] = round(entity['confidence'], 4)
        rows.append(row)
    if rows:
        st.dataframe(rows, hide_index=True, width='stretch')
    elif not entities:
        st.info('No entities detected. This is a model prediction, not proof that the text contains none.')
    else:
        st.info('No entities match the selected filters.')
    st.caption('Offsets are zero-based Unicode code points; ends are exclusive. ' + result['confidence_definition'])
    one, two = st.columns(2)
    stem = 'crf' if family == 'Classical CRF' else 'bilstm_crf'
    one.download_button('Download JSON', json_export(result), f'indicnewsner_{stem}.json', 'application/json', key=f'json_{family}', width='stretch')
    two.download_button('Download CSV', csv_export(result), f'indicnewsner_{stem}.csv', 'text/csv', key=f'csv_{family}', width='stretch')
    st.caption('Downloads include all detected entities, regardless of the table filter.')
    diagnostics = result['diagnostics']
    st.caption(f"{diagnostics['token_count']} tokens · Verified prediction took {diagnostics['inference_time_ms']:.1f} ms, including checksum checks; excludes initial model load. This is not a speed benchmark.")
    if diagnostics['unknown_tokens'] is not None:
        st.caption(f"Vocabulary: {diagnostics['known_tokens']} known / {diagnostics['unknown_tokens']} unknown tokens · Input UNK rate {diagnostics['unknown_rate']:.2%}. {diagnostics.get('device_note', '')}")
    else:
        st.caption('Vocabulary coverage is not applicable to the feature-based classical CRF.')


results = {family: result for family, result in st.session_state.get('results', {}).items()
           if family in predictors}
if results and any(result['text'] != text for result in results.values()):
    st.info('The input has changed. Select Extract Entities to refresh the results.')
elif results:
    if mode == 'Compare Both':
        for column, family in zip(st.columns(2), families):
            with column:
                if family in results:
                    show_result(results[family])
                else:
                    st.info(f'{family} has no result. Resolve its verification or inference error and retry.')
        if all(family in results for family in families):
            comparison = compare_entities(results['Classical CRF'], results['BiLSTM-CRF'])
            st.subheader('Where the predictions differ')
            st.caption('Agreement means identical character boundaries and entity type. Neither model is ground truth; these counts are not accuracy scores.')
            a, b, c = st.columns(3)
            a.metric('Agreed spans', len(comparison['agreed']))
            b.metric('Classical only', len(comparison['classical_only']))
            c.metric('Neural only', len(comparison['neural_only']))
            differences = [dict(Model=family, Entity=e['text'], Type=e['type'], Start=e['start_char'], End=e['end_char'])
                           for key, family in [('classical_only', 'Classical CRF'), ('neural_only', 'BiLSTM-CRF')]
                           for e in comparison[key]]
            if differences:
                st.dataframe(differences, hide_index=True, width='stretch')
            else:
                st.info('Both models predicted the same entity spans for this input.')
            st.download_button('Download comparison JSON', json_export({'text': text, 'results': results, 'comparison': comparison}),
                               'indicnewsner_comparison.json', 'application/json')
    else:
        show_result(results[mode])
else:
    st.info('Choose an example or enter Hindi text, then select Extract Entities.')

with st.expander('Model details · offline evaluation', expanded=False):
    st.caption('Validation results use the same 12,896 benchmark records. They are not measurements of your input or independently verified current-news performance.')
    for family in families:
        st.markdown(f'**{family} · Hindi · PER / ORG / LOC**')
        predictor = predictors.get(family)
        if predictor is None:
            st.warning('Checksum status: unavailable or failed. This model is disabled.')
            continue
        neural = family == 'BiLSTM-CRF'
        st.code(NEURAL_ID if neural else CRF_ID, language=None)
        st.write(f'Training sample: 100,000 clean records · Device: {predictor.device.upper()} · Runtime checksum status: verified')
        st.caption(predictor.device_note)
        values = [0.73818829, 0.73339752, 0.78115319, 0.63522998, 0.78380938] if neural else [0.713769728, 0.709317452, 0.751631012, 0.608512135, 0.767809209]
        st.table([dict(Metric=metric, Value=f'{value:.6f}') for metric, value in zip(
            ['Validation strict micro F1', 'Validation strict macro F1', 'Validation PER F1', 'Validation ORG F1', 'Validation LOC F1'], values)])
        if neural:
            st.caption('Model size: 36.92 MiB · Frozen vocabulary: 94,405 entries · Offline validation UNK rate: 2.54%. Unseen input tokens map to UNK. No neural final-test result is available.')
        else:
            st.caption('Model size: 21.66 MiB · Historical clean-test strict micro F1: 0.745064. This is a separate held-out result, not comparable to neural validation scores. No test files are loaded by this app.')
with st.expander('Limitations and interpretation', expanded=False):
    st.markdown('Current Indian-news performance has not been independently verified. Naamapadam is multi-domain. Organisation names and entity boundaries remain difficult.\n\nBoth models use the same Unicode-aware tokens and sentence boundaries. Raw-text tokenisation may differ from benchmark-supplied tokens; periods in abbreviations and decimals can create extra sentence breaks. Limits: 5,000 Unicode code points and 300 tokens per sentence.\n\nThe CRF uses token features; the BiLSTM-CRF maps exact tokens to a frozen vocabulary. Neural confidence is omitted. CRF marginals are not calibrated entity correctness. Device fallback uses the same neural checkpoint on CPU, never another model. Cross-device bitwise equivalence is not promised.\n\nInvalid I-tags do not start entities. Text stays in session memory. CSV formula-like text is prefixed with an apostrophe; JSON preserves exact text. These two frozen baselines are Hindi only. English NER and Hindi → English translation are separate experimental tools available from the sidebar. IndicBERT is not integrated.')
