"""Unicode-preserving, configurable local features for Hindi CRF."""
import unicodedata as ud


def token_features(token, config):
    if not isinstance(token, str) or not token:
        raise ValueError("Empty/non-string token")
    names = [ud.name(c, "") for c in token]
    categories = [ud.category(c) for c in token]
    latin = any("LATIN" in name for name in names)
    deva = any("DEVANAGARI" in name for name in names)
    nonlatin_letters = any(c.isalpha() and "LATIN" not in name for c, name in zip(token, names))
    f = {"token": token, "length": len(token), "contains_digit": any(c.isdigit() for c in token),
         "numeric": token.isnumeric(), "contains_punctuation": any(c.startswith("P") for c in categories),
         "punctuation": all(c.startswith("P") for c in categories), "alphabetic": token.isalpha()}
    for n in config["prefixes"]:
        f[f"prefix{n}"] = token[:n]
    for n in config["suffixes"]:
        f[f"suffix{n}"] = token[-n:]
    if config["script_features"]:
        f.update(latin=latin, devanagari=deva, mixed_script=latin and deva)
        for category in sorted(set(categories)):
            f[f"unicode_category_{category}"] = True
    if latin and not nonlatin_letters:
        f["latin_lower"] = token.lower()
        f["latin_upper"] = token.isupper()
        f["latin_title"] = token.istitle()
    if config["token_shape"]:
        f["shape"] = "".join("D" if c.isdigit() else "X" if c.isupper() else "x" if "LATIN" in name else "H" if "DEVANAGARI" in name else category[0] for c, name, category in zip(token, names, categories))
    return f


def sentence_features(tokens, config):
    if config.get("gazetteers"):
        raise ValueError("Gazetteers are disabled for the controlled baseline")
    window = config["context_window"]
    if not isinstance(window, int) or window < 0:
        raise ValueError("Invalid context window")
    if any(not isinstance(n, int) or n < 1 for n in config["prefixes"] + config["suffixes"]):
        raise ValueError("Invalid affix length")
    base = [token_features(token, config) for token in tokens]
    result = []
    for i, features in enumerate(base):
        current = dict(features, bias=1.0, BOS=i == 0, EOS=i == len(tokens) - 1)
        for offset in range(-window, window + 1):
            if offset and 0 <= i + offset < len(tokens):
                current.update({f"{offset:+d}:{k}": v for k, v in base[i + offset].items()})
        result.append(current)
    return result
