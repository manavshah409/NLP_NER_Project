"""Shared output schema and exact-span disagreement summaries for the two baselines."""
import time
from src.inference.crf_predictor import CRFPredictor
from src.inference.runtime_verification import ROOT, verify_crf_runtime


def normalise_result(result, family, device, elapsed_ms, known=None, unknown=None):
    result.update(model_family=family, experiment_id=result['model'], language='hi',
                  predicted_labels=list(result['predicted_bio_sequence']))
    count = len(result['tokens'])
    result['diagnostics'] = {
        'token_count': count, 'known_tokens': known, 'unknown_tokens': unknown,
        'unknown_rate': unknown / count if unknown is not None and count else None,
        'inference_time_ms': elapsed_ms, 'device': device,
        'timing_definition': 'Wall time for verified prediction, including checksum checks, tokenisation and span recovery; excludes initial model load. Not a benchmark.',
        'vocabulary_definition': 'Exact frozen neural vocabulary membership; null for feature-based classical CRF.',
    }
    return result


class ClassicalPredictor:
    """Add the shared schema without modifying the frozen classical adapter."""
    def __init__(self, root=ROOT):
        self.root = root
        self.frozen = verify_crf_runtime(root)
        self.predictor = CRFPredictor(root)
        self.device = 'cpu'
        self.device_note = 'Native CRFsuite on CPU.'

    def verify(self):
        return verify_crf_runtime(self.root)

    def predict(self, text):
        started = time.perf_counter()
        self.verify()
        result = self.predictor.predict(text)
        return normalise_result(result, 'Classical CRF', self.device, (time.perf_counter()-started)*1000)


def compare_entities(first, second):
    if first['text'] != second['text']:
        raise ValueError('Comparison requires identical original text.')
    def offsets(result):
        return [(t['text'], t['start_char'], t['end_char']) for t in result['tokens']]
    if offsets(first) != offsets(second):
        raise ValueError('Comparison requires identical token boundaries.')
    a = {(e['start_char'], e['end_char'], e['type']): e for e in first['entities']}
    b = {(e['start_char'], e['end_char'], e['type']): e for e in second['entities']}
    return {'agreed': [a[k] for k in sorted(a.keys() & b.keys())],
            'classical_only': [a[k] for k in sorted(a.keys() - b.keys())],
            'neural_only': [b[k] for k in sorted(b.keys() - a.keys())],
            'definition': 'Exact character boundaries and type; agreement is not measured accuracy.'}
