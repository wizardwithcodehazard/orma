"""
text.py - Structural text helpers: splitting records into sentences, whole-word matching, and the
word pairs a paraphrase is compared to. Nothing here reads meaning; it only cuts text where its
form says to (sentence ends, line breaks, code fences).
"""

import math
import re

_WORD = re.compile(r"[^\W_]+")  # splits like the FTS5 unicode61 tokenizer
_FENCE = re.compile(r"```[ \t]*([\w+#.-]*)[^\n]*\n(.*?)(?:```|\Z)", re.S)
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=\S)")
# A period that ends an initial or a short abbreviation ("J.", "Dr.", "e.g.") does not end the sentence.
_ABBREVIATION = re.compile(r"(?:^|\s)(?:[A-Z][a-z]?|[A-Za-z](?:\.[A-Za-z])+)\.$")


def count_tokens(text: str, chars_per_token: float) -> int:
    """An estimate: characters divided by chars_per_token, rounded up."""
    return math.ceil(len(text) / chars_per_token)


def words(text: str) -> list[str]:
    return [w.casefold() for w in _WORD.findall(text)]


def span_of(text: str, name: str) -> tuple[int, int] | None:
    """(start, end) word positions of name as a whole-word run inside text, else None."""
    haystack, needle = words(text), words(name)
    if not needle:
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i, i + len(needle)
    return None


def made_of(text: str, vocabulary: set[str]) -> bool:
    """Every word of text belongs to vocabulary (a closed word class such as pronouns)."""
    tokens = words(text)
    return bool(tokens) and all(t in vocabulary for t in tokens)


def phrases(text: str) -> list[str]:
    """The words and adjacent word pairs of text, distinct and in order: what a paraphrased name is compared to."""
    found = _WORD.findall(text)
    return list(dict.fromkeys(found + [f"{a} {b}" for a, b in zip(found, found[1:])]))


def _collapse_code(text: str) -> str:
    """A fenced code block becomes one marker line; its text stays in the record, not in the sentences."""
    def marker(match: re.Match) -> str:
        lines = len(match.group(2).strip("\n").splitlines())
        return f"\n[code block{': ' + match.group(1) if match.group(1) else ''}, {lines} lines]\n"
    return _FENCE.sub(marker, text)


def _pieces(sentence: str, max_chars: int) -> list[str]:
    """A sentence longer than max_chars, cut at word boundaries."""
    if len(sentence) <= max_chars:
        return [sentence]
    out, current = [], ""
    for word in sentence.split():
        if current and len(current) + 1 + len(word) > max_chars:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    return out + ([current] if current else [])


def split_sentences(text: str, max_chars: int = 400) -> list[str]:
    """The sentences of a record: split at line breaks and sentence ends, code blocks reduced to a marker, long
    sentences cut to max_chars, whitespace collapsed, pieces without a letter or digit dropped."""
    out = []
    for line in _collapse_code(text).splitlines():
        merged: list[str] = []
        for sentence in _SENTENCE_END.split(line):
            if merged and _ABBREVIATION.search(merged[-1]):
                merged[-1] = f"{merged[-1]} {sentence}"
            else:
                merged.append(sentence)
        for sentence in merged:
            sentence = " ".join(sentence.split())
            if _WORD.search(sentence):
                out.extend(_pieces(sentence, max_chars))
    return out
