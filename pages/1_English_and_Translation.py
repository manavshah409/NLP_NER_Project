"""Optional local language tools; independent of frozen Hindi model loading."""
import streamlit as st
from src.inference.language_bridge import run_language_task
from src.inference.entity_formatter import highlight
from src.inference.exporter import json_export, csv_export

st.set_page_config(page_title='IndicNewsNER · Language tools', page_icon='🌐', layout='wide')
st.title('English NER & Hindi → English')
st.caption('Experimental local tools · Separate pretrained models · No project accuracy claim')
task = st.radio('Choose a tool', ['English entity recognition', 'Hindi → English translation'], horizontal=True)
key = 'english_ner' if task.startswith('English') else 'translate_hi_en'
if key == 'translate_hi_en':
    st.warning('Experimental translation: this model can change names and places. Check every translation against the Hindi source before using it.')
example = ('Meera Sharma visited London and met the Microsoft team.' if key == 'english_ner'
           else 'मीरा शर्मा ने जयपुर में पुस्तक प्रदर्शनी देखी।')
if st.button('Use original example'):
    st.session_state['language_input'] = example
    st.session_state.pop('language_result', None)
text = st.text_area('English text' if key == 'english_ner' else 'Hindi text', key='language_input', height=160)
st.caption(f'{len(text):,} / 5,000 characters. Translation also has a 512-model-token limit. Input is processed locally and is not saved.')
if st.button('Run local tool', type='primary'):
    st.session_state.pop('language_result', None)
    try:
        with st.spinner('Loading the separate model and processing locally…'):
            result = run_language_task(key, text)
        st.session_state['language_result'] = (key, result)
    except (ValueError, RuntimeError) as error:
        st.error(str(error))
saved = st.session_state.get('language_result')
if saved and (saved[0] != key or saved[1]['text'] != text):
    st.info('The input or tool has changed. Run again to refresh the result.')
elif saved:
    result = saved[1]
    if key == 'english_ner':
        st.markdown(highlight(text, result['entities']), unsafe_allow_html=True)
        if result['entities']:
            st.dataframe([{'Entity': e['text'], 'Type': e['type'], 'Original label': e['source_label'],
                           'Start': e['start_char'], 'End': e['end_char']} for e in result['entities']],
                         hide_index=True, width='stretch')
        else:
            st.info('No supported entities detected.')
        csv_result = {'entities': [{k: v for k, v in entity.items() if k != 'source_label'}
                                   for entity in result['entities']]}
        st.download_button('Download CSV', csv_export(csv_result), 'english_entities.csv', 'text/csv')
    else:
        st.subheader('English translation')
        st.text(result['translated_text'])
        st.download_button('Download translation text', result['translated_text'], 'translation_en.txt', 'text/plain')
    st.download_button('Download JSON', json_export(result), 'language_result.json', 'application/json')
    st.caption(result['limitations'])
    st.caption('CPU · Local model checksums verified · Includes model-load time: ' + str(result['elapsed_ms_including_model_load']) + ' ms')
with st.expander('Models and interpretation'):
    st.write('English: spaCy en_core_web_sm 3.8.0. PERSON maps to PER; ORG remains ORG; GPE and LOC map to LOC. Other labels are excluded. English tokenisation is spaCy-native and each token retains original character offsets.')
    st.write('Translation: Helsinki-NLP/opus-mt-hi-en. Hindi → English only. Review machine translations for changed names, omissions and factual errors. No automatic entity alignment is claimed.')
    st.write('These are external pretrained tools, not newly trained project baselines. Their performance has not been evaluated on project data or current Indian news. The Hindi comparison remains on the main page.')
