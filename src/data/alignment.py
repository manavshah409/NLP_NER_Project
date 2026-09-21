"""Align BIO labels to tokenizer word IDs; special/padding tokens are ignored."""
from src.data.bio import LABELS


def align_labels(labels, word_ids, label_all_subtokens=False):
    if any(label not in LABELS for label in labels):
        raise ValueError("Unknown BIO label")
    result, previous = [], None
    for word_id in word_ids:
        if word_id is None:
            result.append(-100)
        else:
            if not isinstance(word_id, int) or not 0 <= word_id < len(labels):
                raise ValueError("Word ID out of range")
            label = labels[word_id]
            if word_id == previous:
                label = "I-" + label[2:] if label.startswith("B-") else label
                result.append(LABELS.index(label) if label_all_subtokens else -100)
            else:
                result.append(LABELS.index(label))
        previous = word_id
    return result
