"""English text frontend: normalisation and sentence-level grapheme-to-phoneme.

The reference repository looks every word up in a lexicon and keeps the first
pronunciation, so a heteronym such as "conduct" never changes with its part of
speech. It also has no rules for times or dates. `text_to_phone_groups` fixes
both; `stock_phones` keeps the original behaviour so the two can be compared.
"""
import re
from string import punctuation

import inflect
import nltk

from .numbers import normalize_numbers

_inflect = inflect.engine()

# Stand-ins that survive tokenisation and are pronounced through _OVERRIDES.
_AM, _PM = "ayyem", "peeyem"

_OVERRIDES = {
    _AM: "EY1 EH1 M".split(),
    _PM: "P IY1 EH1 M".split(),
    # Telugu "andaroo" (everyone), written in Roman script inside an English sentence
    "andaroo": "AA1 N D AH0 R UW1".split(),
}

_CLITICS = {
    "'s": [], "'ll": ["L"], "'re": ["R"], "'ve": ["V"], "'d": ["D"], "'m": ["M"],
    "n't": ["N", "T"],
}
_SIBILANTS = {"S", "Z", "SH", "ZH", "CH", "JH"}
_VOICELESS = {"P", "T", "K", "F", "TH"}

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
)
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?:\s*([AaPp])\.?[Mm]\.?(?![A-Za-z]))?")
_DATE = re.compile(
    r"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({})\b(?:\s*,?\s*(\d{{4}}))?".format(_MONTHS),
    re.IGNORECASE,
)
_SYMBOLS = {"%": " percent ", "&": " and ", "+": " plus ", "@": " at ", "=": " equals "}

_g2p = None


def _get_g2p():
    global _g2p
    if _g2p is None:
        for package, kind in [
            ("averaged_perceptron_tagger_eng", "taggers"),
            ("averaged_perceptron_tagger", "taggers"),
            ("punkt_tab", "tokenizers"),
            ("cmudict", "corpora"),
        ]:
            try:
                nltk.data.find("{}/{}".format(kind, package))
            except LookupError:
                nltk.download(package, quiet=True)
        from g2p_en import G2p

        _g2p = G2p()
    return _g2p


def _expand_time(match):
    hours, minutes, meridiem = int(match.group(1)), int(match.group(2)), match.group(3)
    spoken = _inflect.number_to_words(hours)
    if minutes == 0:
        spoken += " o'clock" if not meridiem else ""
    elif minutes < 10:
        spoken += " oh " + _inflect.number_to_words(minutes)
    else:
        spoken += " " + _inflect.number_to_words(minutes).replace("-", " ")
    if meridiem:
        spoken += " " + (_AM if meridiem.lower() == "a" else _PM)
    return spoken


def _expand_date(match):
    day = _inflect.number_to_words(_inflect.ordinal(int(match.group(1))))
    spoken = "the {} of {}".format(day.replace("-", " "), match.group(2))
    if match.group(3):
        spoken += " " + match.group(3)
    return spoken


def normalize_text(text):
    """Expand times, dates, numbers and symbols into words."""
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("\r", " ").replace("\n", " ")
    text = _TIME.sub(_expand_time, text)
    text = _DATE.sub(_expand_date, text)
    text = normalize_numbers(text)
    for symbol, word in _SYMBOLS.items():
        text = text.replace(symbol, word)
    text = re.sub(r"(?<=[a-zA-Z])-(?=[a-zA-Z])", " ", text).replace("-", " ")
    text = re.sub(r"[\"()\[\]_]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _attach_clitic(groups, clitic, g2p):
    """Join a token such as 's or n't back onto the word it was split from."""
    word, phones = groups[-1]
    merged = word + clitic
    if merged in g2p.cmu:
        groups[-1] = (merged, list(g2p.cmu[merged][0]))
    elif clitic == "'s":
        last = phones[-1]
        suffix = ["IH0", "Z"] if last in _SIBILANTS else ["S"] if last in _VOICELESS else ["Z"]
        groups[-1] = (merged, phones + suffix)
    else:
        groups[-1] = (merged, phones + _CLITICS[clitic])


def corpus_words(text):
    """Word tokens exactly as they are written to the aligner's .lab files."""
    words = re.findall(r"[a-z']+", normalize_text(text))
    return [w.strip("'") for w in words if w.strip("'")]


def text_to_phone_groups(text):
    """Return [(word, phones)] using part-of-speech aware G2P on the whole sentence.

    This mirrors G2p.__call__ so that overrides can be applied and word
    boundaries kept for the variance-adaptor plots.
    """
    g2p = _get_g2p()
    tokens = nltk.word_tokenize(normalize_text(text))
    groups = []
    for word, pos in nltk.pos_tag(tokens):
        if word in _CLITICS and groups:
            _attach_clitic(groups, word, g2p)
            continue
        if word in _OVERRIDES:
            phones = _OVERRIDES[word]
        elif re.search("[a-z]", word) is None:
            if word in ",;:.?!" and groups and groups[-1][1] != ["sp"]:
                groups.append((word, ["sp"]))
            continue
        elif word in g2p.homograph2features:
            pron1, pron2, pos1 = g2p.homograph2features[word]
            phones = pron1 if pos.startswith(pos1) else pron2
        elif word in g2p.cmu:
            phones = g2p.cmu[word][0]
        else:
            phones = g2p.predict(word)
        groups.append((word, list(phones)))
    while groups and groups[-1][1] == ["sp"]:
        groups.pop()
    return groups


def read_lexicon(lex_path):
    """All pronunciations of each word, lower-cased, in file order."""
    lexicon = {}
    with open(lex_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            prons = lexicon.setdefault(parts[0].lower(), [])
            if parts[1:] not in prons:
                prons.append(parts[1:])
    return lexicon


def pronunciations(word, lexicon):
    """Lexicon entries for the aligner, falling back to the neural G2P for unknown words."""
    if word in _OVERRIDES:
        return [_OVERRIDES[word]]
    if word in lexicon:
        return lexicon[word]
    return [[p for p in _get_g2p()(word) if re.fullmatch(r"[A-Z]+[0-2]?", p)]]


def stock_phones(text, lexicon):
    """The reference repository's frontend, kept for the stock-vs-improved comparison."""
    text = text.rstrip(punctuation)
    g2p = _get_g2p()
    phones = []
    for w in re.split(r"([,;.\-\?\!\s+])", text):
        if w.lower() in lexicon:
            phones += lexicon[w.lower()][0]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    return phones.replace("}{", " ").strip("{}").split()
