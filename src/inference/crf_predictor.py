"""Frozen-model runtime verification and thread-safe inference for the local demo."""
import importlib.metadata
import threading
from datetime import datetime, timezone
from pathlib import Path
import yaml
from src.data.bio import LABELS
from src.data.crf_features import sentence_features
from src.models.crf_model import CRFModel
from src.utils.io import digest, file_hash, read_json
from src.inference.text_preprocessor import tokenize, sentence_ranges
from src.inference.entity_formatter import format_entities

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "crf_100k_seed42_24dd05e84a26"
FREEZE_PATH = "reports/crf/100k_baseline_freeze_manifest.json"
LIMITATIONS = "Hindi CRF; Naamapadam is multi-domain, not verified current Indian-news performance. Raw-text tokenisation differs from benchmark tokenisation. No English support is claimed."
EXAMPLES = {
    "Person and place": "मीरा शर्मा ने जयपुर में पुस्तक प्रदर्शनी देखी।",
    "Organisation and place": "भारतीय स्टेट बैंक ने भोपाल में नई शाखा खोली।",
    "Multiple entities": "राहुल वर्मा और नेहा गुप्ता सोमवार को मुंबई से पुणे गए।",
    "Punctuation and lines": "‘अनिता राव’, लखनऊ पहुंचीं।\nउन्होंने कहा: बैठक कल होगी!",
    "Everyday sentence": "आज मौसम सुहावना है और हल्की हवा चल रही है।",
}


def verify_runtime(root=ROOT):
    """Check the frozen runtime subset; deliberately never open dataset/test files."""
    root = Path(root)
    envelope = read_json(root / FREEZE_PATH)
    payload = envelope["payload"]
    if digest(payload) != envelope["payload_sha256"] or payload["experiment_id"] != EXPERIMENT:
        raise ValueError("Baseline freeze manifest is invalid or identifies another model.")
    model_dir = f"models/crf/{EXPERIMENT}"
    required = [f"{model_dir}/{name}" for name in ("model.crfsuite", "resolved_config.yaml", "environment.json")]
    required += ["src/data/bio.py", "src/data/crf_features.py", "src/models/crf_model.py", "src/utils/io.py",
                 "src/data/__init__.py", "src/models/__init__.py", "src/utils/__init__.py", "src/__init__.py"]
    for name in required:
        if name not in payload["files"] or file_hash(root / name) != payload["files"][name]:
            raise ValueError(f"Frozen model/runtime checksum failed: {name}. Restore the original artifact; no fallback model will be loaded.")
    for package, version in payload["environment"]["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Frozen dependency version differs: {package}. Restore the pinned environment.")
    if list(payload["label_mapping"].values()) != list(LABELS):
        raise ValueError("Frozen label mapping differs")
    config = yaml.safe_load((root / model_dir / "resolved_config.yaml").read_text())
    if config != payload["config"]:
        raise ValueError("Frozen configuration differs")
    return payload


class CRFPredictor:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.frozen = verify_runtime(self.root)
        self.model = CRFModel(self.frozen["config"]["model"]).load(self.root / self.frozen["model_dir"] / "model.crfsuite")
        self.lock = threading.Lock()

    def predict(self, text):
        tokens = tokenize(text)
        sentences = list(sentence_ranges(text, tokens))
        verify_runtime(self.root)
        labels, entities, marginals = [], [], []
        with self.lock:
            for start, end in sentences:
                current = tokens[start:end]
                predicted = self.model.predict(sentence_features([t["text"] for t in current], self.frozen["config"]["features"]))
                probabilities = [self.model.tagger.marginal(label, i) for i, label in enumerate(predicted)]
                entities.extend(format_entities(text, current, predicted, probabilities, start))
                labels.extend(predicted)
                marginals.extend(probabilities)
        for entity in entities:
            if text[entity["start_char"]:entity["end_char"]] != entity["text"]:
                raise ValueError("Entity offsets could not be recovered")
        return {"text": text, "model": EXPERIMENT, "dataset_version": self.frozen["dataset_version"],
                "processed_at": datetime.now(timezone.utc).isoformat(), "entities": entities,
                "tokens": [dict(t, index=i, bio=labels[i], predicted_tag_marginal=marginals[i]) for i, t in enumerate(tokens)],
                "predicted_bio_sequence": labels,
                "offset_convention": "Zero-based Python Unicode code-point offsets; character and token ends are exclusive. Token indices are global across the input.",
                "confidence_definition": "Arithmetic mean of CRFsuite marginal probabilities for the entity's predicted token tags. Not calibrated entity correctness or factual truth.",
                "limitations": LIMITATIONS}
