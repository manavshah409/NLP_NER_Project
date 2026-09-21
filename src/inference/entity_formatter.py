"""Recover typed spans and safely highlight the original text by offsets."""
import html
import math
from src.data.bio import strict_spans

COLORS = {"PER": ("#dbeafe", "#1e40af"), "ORG": ("#ffedd5", "#9a3412"), "LOC": ("#dcfce7", "#166534")}


def mean_marginal(values):
    if not values or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Invalid CRFsuite marginal probabilities")
    return sum(values) / len(values)


def format_entities(text, tokens, labels, marginals, token_offset=0):
    if not len(tokens) == len(labels) == len(marginals):
        raise ValueError("Token, prediction and marginal lengths differ")
    entities = []
    for start, end, kind in strict_spans(labels):
        a, b = tokens[start]["start_char"], tokens[end-1]["end_char"]
        entities.append({"text": text[a:b], "type": kind, "start_char": a, "end_char": b,
                         "start_token": start+token_offset, "end_token": end+token_offset,
                         "bio_sequence": labels[start:end], "confidence": mean_marginal(marginals[start:end])})
    return entities


def highlight(text, entities):
    pieces, cursor = [], 0
    for entity in sorted(entities, key=lambda e: e["start_char"]):
        start, end = entity["start_char"], entity["end_char"]
        if not cursor <= start < end <= len(text) or text[start:end] != entity["text"]:
            raise ValueError("Invalid or overlapping entity offsets")
        background, foreground = COLORS[entity["type"]]
        pieces.append(html.escape(text[cursor:start]))
        pieces.append(f'<mark style="background:{background};color:{foreground};padding:2px 3px;border-radius:4px" title="{entity["type"]}">{html.escape(text[start:end])}</mark>')
        cursor = end
    pieces.append(html.escape(text[cursor:]))
    return '<div class="entity-output" style="white-space:pre-wrap;overflow-wrap:anywhere;font-size:1.18rem;line-height:2.1">' + "".join(pieces) + "</div>"
