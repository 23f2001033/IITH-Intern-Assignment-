"""Rule-based Telugu text frontend: number normalisation and grapheme-to-phoneme.

Telugu is an abugida with a close spelling-to-sound mapping, so phonemes can be
derived from the script without a trained G2P model:
  * a consonant carries the inherent vowel "a" unless a matra or virama follows
  * a matra replaces the inherent vowel
  * a virama removes it, which is how conjuncts and geminates are written
  * the anusvara becomes the nasal of the following consonant's place
"""
import re
import unicodedata

VIRAMA = "్"
ANUSVARA = "ం"
CANDRABINDU = "ఁ"
VISARGA = "ః"

_VOWELS = {
    "అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"], "ఉ": ["u"], "ఊ": ["uu"],
    "ఋ": ["r", "u"], "ౠ": ["r", "uu"], "ఎ": ["e"], "ఏ": ["ee"], "ఐ": ["ai"],
    "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"],
}

_MATRAS = {
    "ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"], "ూ": ["uu"],
    "ృ": ["r", "u"], "ౄ": ["r", "uu"], "ె": ["e"], "ే": ["ee"], "ై": ["ai"],
    "ొ": ["o"], "ో": ["oo"], "ౌ": ["au"],
}

_CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "c", "ఛ": "ch", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "tt", "ఠ": "tth", "డ": "dd", "ఢ": "ddh", "ణ": "nn",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "r", "ల": "l", "ళ": "ll", "వ": "v",
    "శ": "sh", "ష": "ss", "స": "s", "హ": "h",
    "ౘ": "c", "ౙ": "j",
}

_NASAL_BEFORE = {
    "k": "ng", "kh": "ng", "g": "ng", "gh": "ng",
    "c": "ny", "ch": "ny", "j": "ny", "jh": "ny",
    "tt": "nn", "tth": "nn", "dd": "nn", "ddh": "nn",
    "t": "n", "th": "n", "d": "n", "dh": "n", "n": "n",
}

valid_symbols = sorted(
    {p for v in _VOWELS.values() for p in v}
    | set(_CONSONANTS.values())
    | set(_NASAL_BEFORE.values())
)

_UNITS = [
    "సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది",
    "తొమ్మిది", "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను",
    "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పందొమ్మిది",
]
_TENS = {
    2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై",
    8: "ఎనభై", 9: "తొంభై",
}

_TELUGU_WORD = re.compile(r"[ఀ-౿]+")
_TOKEN = re.compile(r"[ఀ-౿]+|[,;:.?!]+")


def number_to_words(n):
    """Cardinal number name, up to 99,99,999; larger numbers are read digit by digit."""
    if n < 20:
        return _UNITS[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        return _TENS[tens] + (" " + _UNITS[unit] if unit else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        if hundreds == 1:
            return "వంద" if rest == 0 else "నూట " + number_to_words(rest)
        if rest == 0:
            return _UNITS[hundreds] + " వందలు"
        return _UNITS[hundreds] + " వందల " + number_to_words(rest)
    if n < 100000:
        thousands, rest = divmod(n, 1000)
        if thousands == 1:
            head = "వెయ్యి"
        else:
            head = number_to_words(thousands) + (" వేలు" if rest == 0 else " వేల")
        return head if rest == 0 else head + " " + number_to_words(rest)
    if n < 10000000:
        lakhs, rest = divmod(n, 100000)
        if lakhs == 1:
            head = "లక్ష"
        else:
            head = number_to_words(lakhs) + (" లక్షలు" if rest == 0 else " లక్షల")
        return head if rest == 0 else head + " " + number_to_words(rest)
    return " ".join(_UNITS[int(d)] for d in str(n))


def ordinal_to_words(n):
    """12 -> పన్నెండవ: the last word of the cardinal takes the ordinal suffix."""
    words = number_to_words(n)
    if words.endswith("ై"):
        return words[:-1] + "య్యవ"
    if words[-1] in "ుి":
        return words[:-1] + "వ"
    return words + "వ"


def _expand_time(match):
    hours, minutes = int(match.group(1)), int(match.group(2))
    locative = bool(match.group(3))
    if minutes == 0:
        return number_to_words(hours) + (" గంటలకు" if locative else " గంటలు")
    return "{} గంటల {} {}".format(
        number_to_words(hours),
        number_to_words(minutes),
        "నిమిషాలకు" if locative else "నిమిషాలు",
    )


def normalize_text(text):
    """Expand times, ordinals and numbers into Telugu words."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("﻿", "").replace("‌", "").replace("‍", "")
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b\s*(గంటలకు)?", _expand_time, text)
    text = re.sub(r"(\d+)వ(?![ా-్])", lambda m: ordinal_to_words(int(m.group(1))), text)
    text = re.sub(r"\d+", lambda m: " " + number_to_words(int(m.group(0))) + " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_to_phones(word):
    phones = []
    chars = unicodedata.normalize("NFC", word)
    i = 0
    while i < len(chars):
        ch = chars[i]
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if ch in _CONSONANTS:
            phones.append(_CONSONANTS[ch])
            if nxt == VIRAMA:
                i += 2
            elif nxt in _MATRAS:
                phones.extend(_MATRAS[nxt])
                i += 2
            else:
                phones.append("a")
                i += 1
        elif ch in _VOWELS:
            phones.extend(_VOWELS[ch])
            i += 1
        elif ch == ANUSVARA:
            phones.append(_NASAL_BEFORE.get(_CONSONANTS.get(nxt), "m"))
            i += 1
        elif ch == VISARGA:
            phones.append("h")
            i += 1
        else:
            # candrabindu, nukta and other marks have no phone of their own
            i += 1
    return phones


def corpus_words(text):
    """Word tokens exactly as they are written to the aligner's .lab files."""
    return _TELUGU_WORD.findall(normalize_text(text))


def text_to_phone_groups(text):
    """Return [(word, phones)]; punctuation becomes a short pause ("sp")."""
    groups = []
    for token in _TOKEN.findall(normalize_text(text)):
        if _TELUGU_WORD.fullmatch(token):
            groups.append((token, word_to_phones(token)))
        elif groups and groups[-1][1] != ["sp"]:
            groups.append((token, ["sp"]))
    while groups and groups[-1][1] == ["sp"]:
        groups.pop()
    return groups
