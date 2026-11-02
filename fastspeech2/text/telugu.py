"""Rule-based Telugu front-end: text normalization, numbers -> words, graphemes -> phonemes.

Telugu script is close to phonemic, so each akshara maps to phones directly:
consonant (+ inherent "a" unless a vowel sign or virama follows), independent vowels,
anusvara -> nasal of the next consonant's place, visarga -> "h".
Phone names are lowercase ASCII so they never clash with ARPAbet (uppercase).
"""
import re

VOWELS = {  # independent vowel letters
    "అ": "a", "ఆ": "aa", "ఇ": "i", "ఈ": "ii", "ఉ": "u", "ఊ": "uu", "ఋ": "r u", "ౠ": "r uu",
    "ఎ": "e", "ఏ": "ee", "ఐ": "ai", "ఒ": "o", "ఓ": "oo", "ఔ": "au",
}
SIGNS = {  # dependent vowel signs (matras)
    "ా": "aa", "ి": "i", "ీ": "ii", "ు": "u", "ూ": "uu", "ృ": "r u", "ౄ": "r uu",
    "ె": "e", "ే": "ee", "ై": "ai", "ొ": "o", "ో": "oo", "ౌ": "au",
}
CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "c", "ఛ": "ch", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "tt", "ఠ": "tth", "డ": "dd", "ఢ": "ddh", "ణ": "nn",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "r", "ల": "l", "వ": "v",
    "శ": "sh", "ష": "ss", "స": "s", "హ": "h", "ళ": "ll",
}
VIRAMA, ANUSVARA, VISARGA = "్", "ం", "ః"
# anusvara takes the place of articulation of the following consonant
NASAL_FOR = {}
for group, nasal in (("కఖగఘఙ", "ng"), ("చఛజఝఞ", "ny"), ("టఠడఢణ", "nn"), ("తథదధన", "n"), ("పఫబభమ", "m")):
    for ch in group:
        NASAL_FOR[ch] = nasal

PHONES = sorted({p for v in (VOWELS, SIGNS, CONSONANTS) for s in v.values() for p in s.split()} | {"m", "h"})

ONES = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
        "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పందొమ్మిది"]
TENS = ["", "", "ఇరవై", "ముప్పై", "నలభై", "యాభై", "అరవై", "డెబ్బై", "ఎనభై", "తొంభై"]
HUNDREDS = ["", "నూట", "రెండు వందల", "మూడు వందల", "నాలుగు వందల", "ఐదు వందల", "ఆరు వందల", "ఏడు వందల", "ఎనిమిది వందల", "తొమ్మిది వందల"]
EXACT_HUNDREDS = ["", "వంద", "రెండు వందలు", "మూడు వందలు", "నాలుగు వందలు", "ఐదు వందలు", "ఆరు వందలు", "ఏడు వందలు", "ఎనిమిది వందలు", "తొమ్మిది వందలు"]


def number_to_words(n):
    """0..99999 -> Telugu words (spoken cardinal form)."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        h, r = divmod(n, 100)
        return EXACT_HUNDREDS[h] if r == 0 else HUNDREDS[h] + " " + number_to_words(r)
    th, r = divmod(n, 1000)
    head = "వెయ్యి" if th == 1 and r == 0 else "వెయ్యి" if th == 1 else number_to_words(th) + (" వేలు" if r == 0 else " వేల")
    return head if r == 0 else head + " " + number_to_words(r)


def _ordinal(words):
    # 12వ -> పన్నెండవ : drop the final vowel sign "ు"/"ి" of the last word and add "వ"
    return re.sub(r"[ుి]?$", "", words) + "వ"


def normalize(text):
    text = text.replace("﻿", "").replace("‌", "").replace("‍", "")
    text = re.sub(r"[A-Za-z]+", " ", text)  # stray Latin letters in the corpus (e.g. "o")
    # a space before a vowel sign / virama / anusvara splits one akshara in two; glue it back
    text = re.sub(r"\s+(?=[ా-్ౕౖంః])", "", text)
    text = re.sub(r"(\d+):(\d+)", lambda m: number_to_words(int(m.group(1))) + " " + number_to_words(int(m.group(2))), text)
    text = re.sub(r"(\d+)వ", lambda m: _ordinal(number_to_words(int(m.group(1)))), text)
    text = re.sub(r"\d+", lambda m: number_to_words(int(m.group())), text)
    text = re.sub(r"[\"“”'‘’]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text):
    """Normalized text -> list of Telugu words (punctuation removed)."""
    return re.findall(r"[ఀ-౿]+", normalize(text))


def word_to_phones(word):
    phones, chars = [], list(word)
    for i, ch in enumerate(chars):
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if ch in CONSONANTS:
            phones += CONSONANTS[ch].split()
            if nxt not in SIGNS and nxt != VIRAMA:
                phones.append("a")  # inherent vowel
        elif ch in VOWELS:
            phones += VOWELS[ch].split()
        elif ch in SIGNS:  # after a consonant, or orphaned: either way it is that vowel
            phones += SIGNS[ch].split()
        elif ch == ANUSVARA:
            phones.append(NASAL_FOR.get(nxt, "m"))
        elif ch == VISARGA:
            phones.append("h")
    return phones


if __name__ == "__main__":
    assert word_to_phones("తెలుగు") == ["t", "e", "l", "u", "g", "u"]
    assert word_to_phones("పట్టు") == ["p", "a", "tt", "tt", "u"]
    assert word_to_phones("అంటే") == ["a", "nn", "tt", "ee"]
    assert word_to_phones("సంబంధం") == ["s", "a", "m", "b", "a", "n", "dh", "a", "m"]
    assert number_to_words(2026) == "రెండు వేల ఇరవై ఆరు"
    assert normalize("12వ తేదీ 10:45 గంటలకు") == "పన్నెండవ తేదీ పది నలభై ఐదు గంటలకు"
    assert normalize("వచ్చ ింది") == "వచ్చింది"
    print(PHONES)
    print("ok")
