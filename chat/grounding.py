"""Checks that an answer says only what its retrieved sources say.

A language model, or the team's retrieval service, may reword a source. This
module decides whether the reworded answer still stays inside the source. It
is deliberately strict: when in doubt, Harbor shows the source text itself
instead of the reworded answer.

An answer is supported when, for every sentence:

* each number in it (1, 3, 72, 100) also appears in the sources, and
* almost all of its content words (four letters or more, ignoring common
  words, with simple plural/tense endings removed) also appear in the sources.
"""

from __future__ import annotations

import re

# Words too common to show that a sentence came from a source.
_STOPWORDS = {
    "about", "above", "after", "again", "also", "always", "another", "because", "been",
    "before", "being", "below", "between", "both", "could", "does", "doing", "down",
    "during", "each", "either", "else", "even", "every", "first", "from", "have", "having",
    "here", "into", "just", "keep", "know", "like", "make", "many", "more", "most", "much",
    "must", "need", "never", "only", "other", "over", "please", "really", "same", "should",
    "some", "still", "such", "sure", "than", "that", "their", "them", "then", "there",
    "these", "they", "thing", "things", "this", "those", "through", "under", "until",
    "very", "want", "well", "were", "what", "when", "where", "which", "while", "will",
    "with", "within", "without", "would", "your", "yours", "yourself", "okay", "here's",
    "important", "remember", "note", "following", "recommended", "recommend", "advice",
    "answer", "question", "source", "sources", "harbor", "according", "says", "said",
}

# Unsupported content words allowed per sentence: max(1, ratio * words).
_ALLOWED_MISS_RATIO = 0.25

_WORD = re.compile(r"[a-z]{4,}")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _content_words(text: str) -> list[str]:
    text = text.lower().replace("’", "'")
    return [_stem(w) for w in _WORD.findall(text) if w not in _STOPWORDS]


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


def sentences(text: str) -> list[str]:
    return [part.strip(" -*•\t") for part in _SENTENCE_SPLIT.split(text or "") if part.strip(" -*•\t")]


def unsupported_sentences(answer: str, source_texts: list[str]) -> list[str]:
    """Return the sentences of ``answer`` that the sources do not back up."""
    joined = "\n".join(t for t in source_texts if t)
    vocabulary = set(_content_words(joined))
    source_numbers = _numbers(joined)
    bad = []
    for sentence in sentences(answer):
        if _numbers(sentence) - source_numbers:
            bad.append(sentence)
            continue
        words = _content_words(sentence)
        if not words:
            continue
        missing = [w for w in words if w not in vocabulary]
        allowed = max(1, int(len(words) * _ALLOWED_MISS_RATIO))
        if len(missing) > allowed:
            bad.append(sentence)
    return bad


def is_supported(answer: str, source_texts: list[str]) -> bool:
    """True when the answer has content and every sentence is backed by the sources."""
    if not answer or not answer.strip():
        return False
    if not any(t and t.strip() for t in source_texts):
        return False
    if not _content_words(answer):
        return False
    return not unsupported_sentences(answer, source_texts)
