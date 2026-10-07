"""Raw-text adapter for the exact frozen neural checkpoint; no evaluation data access."""
from datetime import datetime, timezone
from pathlib import Path
import threading
import time
import torch
from src.models.bilstm_crf import BiLSTM_CRF
from src.data.bilstm_vocabulary import Vocabulary, UNK_ID
from src.data.bio import strict_spans
from src.inference.text_preprocessor import tokenize, sentence_ranges
from src.inference.runtime_verification import ROOT, NEURAL_ID, LABELS, verify_neural_runtime
from src.inference.comparison import normalise_result


class BiLSTMCRFPredictor:
    def __init__(self, root=ROOT, device=None):
        self.root = Path(root)
        self.frozen = verify_neural_runtime(root)
        directory = self.root / 'models' / 'bilstm_crf' / NEURAL_ID
        self.vocabulary = Vocabulary.load(directory / 'vocabulary.json')
        config = self.frozen['config']['model']
        self.model = BiLSTM_CRF(
            vocab_size=len(self.vocabulary), num_tags=len(LABELS),
            embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'],
            num_lstm_layers=config['num_lstm_layers'], bidirectional=config['bidirectional'],
            dropout=config['dropout'], padding_idx=0, use_crf=config['use_crf'])
        state = torch.load(directory / 'best_model.pt', map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.model.requires_grad_(False)
        if device not in (None, 'cpu', 'mps'):
            raise ValueError('Only CPU and MPS inference are supported.')
        requested = device or 'mps'
        available = torch.backends.mps.is_available()
        self.device = 'mps' if requested == 'mps' and available else 'cpu'
        self.device_note = ('MPS acceleration.' if self.device == 'mps' else
                            'CPU selected explicitly.' if device == 'cpu' else 'MPS unavailable; using CPU with the same frozen weights.')
        self.lock = threading.Lock()
        try:
            self.model.to(self.device)
            # Exercise packed LSTM and CRF operations before accepting user input.
            with torch.inference_mode():
                probe = torch.full((1, 2), UNK_ID, dtype=torch.long, device=self.device)
                self.model.decode(probe, mask=torch.ones_like(probe, dtype=torch.bool))
        except (NotImplementedError, RuntimeError) as error:
            if self.device != 'mps' or not self._unsupported_mps(error):
                raise
            self.model.to('cpu')
            self.device = 'cpu'
            self.device_note = 'MPS lacks a required operation; using CPU with the same frozen weights.'

    @staticmethod
    def _unsupported_mps(error):
        message = str(error).lower()
        return 'mps' in message and any(word in message for word in ('not implemented', 'not supported', 'unsupported'))

    def verify(self):
        return verify_neural_runtime(self.root)

    def _decode(self, sequences):
        # Batches have at most 16 sequences, each already limited to 300 tokens.
        paths = []
        for start in range(0, len(sequences), 16):
            current = sequences[start:start+16]
            ids = torch.zeros((len(current), max(map(len, current))), dtype=torch.long, device=self.device)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for row, sequence in enumerate(current):
                ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=self.device)
                mask[row, :len(sequence)] = True
            paths.extend(self.model.decode(ids, mask=mask))
        return paths

    def predict(self, text):
        started = time.perf_counter()
        tokens = tokenize(text)
        ranges = list(sentence_ranges(text, tokens))
        self.verify()
        encoded = self.vocabulary.encode([t['text'] for t in tokens])
        sequences = [encoded[start:end] for start, end in ranges]
        with self.lock, torch.inference_mode():
            try:
                paths = self._decode(sequences)
            except (NotImplementedError, RuntimeError) as error:
                if self.device != 'mps' or not self._unsupported_mps(error):
                    raise
                self.model.to('cpu')
                self.device = 'cpu'
                self.device_note = 'MPS lacks a required operation; using CPU with the same frozen weights.'
                paths = self._decode(sequences)
            if self.device == 'mps':
                torch.mps.synchronize()
        if len(paths) != len(ranges):
            raise ValueError('Neural decoder returned an unexpected number of sentences.')
        labels, entities = [], []
        for (start, end), path in zip(ranges, paths):
            if len(path) != end-start or any(i not in range(len(LABELS)) for i in path):
                raise ValueError('Neural decoder returned invalid tag indices or sequence lengths.')
            predicted = [LABELS[i] for i in path]
            labels.extend(predicted)
            # Reconstruct each sentence separately; I-tags cannot cross sentence boundaries.
            for a, b, kind in strict_spans(predicted):
                first, last = tokens[start+a], tokens[start+b-1]
                entity = {'text': text[first['start_char']:last['end_char']], 'type': kind,
                          'start_char': first['start_char'], 'end_char': last['end_char'],
                          'start_token': start+a, 'end_token': start+b,
                          'bio_sequence': predicted[a:b], 'confidence': None}
                assert text[entity['start_char']:entity['end_char']] == entity['text']
                entities.append(entity)
        unknown = sum(t['text'] not in self.vocabulary.token2id for t in tokens)
        result = {'text': text, 'model': NEURAL_ID, 'dataset_version': self.frozen['dataset_version'],
                  'processed_at': datetime.now(timezone.utc).isoformat(), 'entities': entities,
                  'tokens': [dict(t, index=i, bio=labels[i], token_id=encoded[i]) for i, t in enumerate(tokens)],
                  'predicted_bio_sequence': labels,
                  'offset_convention': 'Zero-based Python Unicode code-point offsets; character and token ends are exclusive. Token indices are global across the input.',
                  'confidence_definition': 'Not provided: Viterbi sequence scores are not calibrated entity probabilities.',
                  'limitations': 'Hindi only. Current-news performance is unverified. Raw-text tokenisation differs from benchmark tokens; unseen words use UNK. No neural final-test result is claimed.'}
        result = normalise_result(result, 'BiLSTM-CRF', self.device, (time.perf_counter()-started)*1000,
                                  len(tokens)-unknown, unknown)
        result['diagnostics']['device_note'] = self.device_note
        return result
