"""Deterministic Unicode tokens with offsets into the untouched input."""
import unicodedata

MAX_CHARACTERS = 5000
MAX_SENTENCE_TOKENS = 300


def tokenize(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter some Hindi text before extracting entities.")
    if len(text) > MAX_CHARACTERS:
        raise ValueError(f"Input exceeds the {MAX_CHARACTERS:,}-character limit. Please shorten it.")
    tokens, i = [], 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        start = i
        if unicodedata.category(text[i])[0] in "LMN":
            i += 1
            while i < len(text) and (unicodedata.category(text[i])[0] in "LMN" or text[i] in "\u200c\u200d"):
                i += 1
        else:
            i += 1
        tokens.append({"text": text[start:i], "start_char": start, "end_char": i})
    return tokens


def sentence_ranges(text, tokens):
    """Split after terminal punctuation or line breaks; ranges are token-exclusive."""
    start = 0
    for i, token in enumerate(tokens):
        gap = text[token["end_char"]:tokens[i+1]["start_char"]] if i+1 < len(tokens) else ""
        if token["text"] in {"।", "॥", ".", "!", "?", "！", "？"} or "\n" in gap or "\r" in gap or i == len(tokens)-1:
            if i+1-start > MAX_SENTENCE_TOKENS:
                raise ValueError(f"A sentence exceeds {MAX_SENTENCE_TOKENS} tokens. Please use shorter sentences.")
            yield start, i+1
            start = i+1
