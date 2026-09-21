"""Separate strict raw BIO interpretation from explicit derived repairs."""
LABELS = ("O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC")
TYPES = ("PER", "ORG", "LOC")


def validate(tokens, labels):
    errors = []
    if tokens is None or labels is None:
        return [{"category": "missing_or_null_fields"}]
    if not isinstance(tokens, list) or not isinstance(labels, list):
        return [{"category": "invalid_field_type"}]
    if len(tokens) != len(labels):
        errors.append({"category": "length_mismatch"})
    if not tokens:
        errors.append({"category": "empty_record"})
    previous = "O"
    for i, label in enumerate(labels):
        if label not in LABELS:
            errors.append({"position": i, "category": "unknown_label", "label": label})
        elif label.startswith("I-"):
            if previous == "O" or previous not in LABELS:
                errors.append({"position": i, "category": "orphan_I", "label": label})
            elif previous[2:] != label[2:]:
                errors.append({"position": i, "category": "type_mismatch_I", "label": label})
        previous = label
    for i, token in enumerate(tokens):
        if not isinstance(token, str) or not token:
            errors.append({"position": i, "category": "invalid_token"})
    return errors


def normalise(tokens, labels):
    errors = validate(tokens, labels)
    if any(e["category"] not in {"empty_record", "orphan_I", "type_mismatch_I"} for e in errors):
        raise ValueError(f"Cannot normalise invalid record: {errors}")
    result = list(labels)
    repairs = []
    previous = "O"
    for i, label in enumerate(result):
        if label.startswith("I-") and (previous == "O" or previous[2:] != label[2:]):
            fixed = "B-" + label[2:]
            repairs.append({"token_position": i, "token_text": tokens[i], "original_label": label,
                            "derived_label": fixed, "error_category": "orphan_I" if previous == "O" else "type_mismatch_I"})
            result[i] = fixed
        previous = result[i]
    return list(tokens), result, repairs


def strict_spans(labels):
    """Return (start, exclusive_end, type); invalid I never starts an entity."""
    spans = []
    start = None
    kind = None
    for i, label in enumerate(list(labels) + ["O"]):
        if label not in LABELS:
            raise ValueError(f"Unknown label: {label}")
        continues = start is not None and label == "I-" + kind
        if start is not None and not continues:
            spans.append((start, i, kind))
            start = kind = None
        if label.startswith("B-"):
            start, kind = i, label[2:]
    return spans
