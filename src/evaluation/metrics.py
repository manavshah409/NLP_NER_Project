"""Strict typed span matching; all PER/ORG/LOC classes included in macro F1."""
from collections import Counter
from src.data.bio import TYPES, strict_spans


def prf(tp, predicted, gold):
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "true_positive": tp, "predicted": predicted, "support": gold}


class StrictMetrics:
    def __init__(self):
        self.tp, self.predicted, self.gold = Counter(), Counter(), Counter()
        self.correct = self.tokens = 0

    def add(self, gold, predicted):
        if len(gold) != len(predicted):
            raise ValueError("Prediction length mismatch")
        g, p = set(strict_spans(gold)), set(strict_spans(predicted))
        self.tp.update(s[2] for s in g & p)
        self.predicted.update(s[2] for s in p)
        self.gold.update(s[2] for s in g)
        self.correct += sum(a == b for a, b in zip(gold, predicted))
        self.tokens += len(gold)

    def result(self):
        classes = {k: prf(self.tp[k], self.predicted[k], self.gold[k]) for k in TYPES}
        micro = prf(sum(self.tp.values()), sum(self.predicted.values()), sum(self.gold.values()))
        return {"strict_entity_precision": micro["precision"], "strict_entity_recall": micro["recall"],
                "strict_entity_micro_f1": micro["f1"], "strict_entity_macro_f1": sum(v["f1"] for v in classes.values()) / 3,
                "per_class": classes, "token_accuracy": self.correct / self.tokens if self.tokens else 0.0,
                "tokens": self.tokens, "policy": "Only B starts a span; exact boundaries and type; macro includes all 3 types"}
