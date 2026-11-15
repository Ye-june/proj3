"""Clean and preprocess dataset documents before they are loaded (story 7).

Every way documents enter Harbor (the dataset file today, uploads or scraping
later) goes through ``chat.knowledge``, which calls :func:`preprocess` before
validation. So text is cleaned once, in one place, before retrieval sees it.

Cleaning steps for ``title``, ``body`` and ``source``:

* Unicode NFKC normalization (non-breaking spaces, full-width letters, ligatures).
* Remove ``<script>``/``<style>`` blocks and HTML tags; block tags become paragraph breaks.
* Decode HTML entities (``&amp;`` -> ``&``).
* Straighten curly quotes so keyword matching and grounding see one form.
* Drop invisible control characters.
* Collapse runs of spaces and blank lines, keeping paragraph breaks in ``body``.

Keywords get the same text cleaning and duplicates are removed. A document
whose body repeats an earlier document's body is dropped, so one passage
cannot crowd out others in retrieval.
"""

from __future__ import annotations

import html
import re
import unicodedata

_SCRIPT_OR_STYLE = re.compile(r"<(script|style)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_BLOCK_TAG = re.compile(r"</?(?:p|div|br|li|ul|ol|h[1-6]|tr|table|section|article)\b[^>]*>", re.IGNORECASE)
# Only real tags: "<" followed by a letter or "/". Text like "a < b > c" is kept.
_TAG = re.compile(r"</?[a-zA-Z][^<>]*>")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

_TEXT_FIELDS = ("title", "body", "source")


def clean_text(text: str, keep_paragraphs: bool = False) -> str:
    """Return ``text`` with markup, odd characters and extra whitespace removed."""
    text = unicodedata.normalize("NFKC", text)
    text = _SCRIPT_OR_STYLE.sub(" ", text)
    text = _BLOCK_TAG.sub("\n\n", text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)  # entities like &nbsp; decode to characters NFKC fixes
    text = text.translate(_QUOTES)
    text = "".join(ch for ch in text if ch == "\n" or not unicodedata.category(ch).startswith("C"))
    if not keep_paragraphs:
        return " ".join(text.split())
    paragraphs = (" ".join(part.split()) for part in _PARAGRAPH_BREAK.split(text))
    return "\n\n".join(p for p in paragraphs if p)


def preprocess(data) -> tuple[object, list[str]]:
    """Clean every document in ``data`` (the parsed dataset file).

    Returns (cleaned data, list of human-readable changes). Anything with the
    wrong shape is passed through untouched so validation can report it.
    """
    if not isinstance(data, dict) or not isinstance(data.get("documents"), list):
        return data, []
    changes: list[str] = []
    documents: list = []
    seen_bodies: dict[str, str] = {}
    for index, item in enumerate(data["documents"]):
        if not isinstance(item, dict):
            documents.append(item)
            continue
        doc = dict(item)
        name = doc["id"] if isinstance(doc.get("id"), str) else f"#{index + 1}"
        cleaned_fields = []
        for field in _TEXT_FIELDS:
            value = doc.get(field)
            if isinstance(value, str):
                cleaned = clean_text(value, keep_paragraphs=field == "body")
                if cleaned != value.strip():
                    cleaned_fields.append(field)
                doc[field] = cleaned
        keywords = doc.get("keywords")
        if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
            unique = list(dict.fromkeys(k for k in (clean_text(k).lower() for k in keywords) if k))
            if unique != keywords:
                cleaned_fields.append("keywords")
            doc["keywords"] = unique
        if cleaned_fields:
            changes.append(f'document "{name}": cleaned {", ".join(cleaned_fields)}')

        body = doc.get("body")
        if isinstance(body, str) and body:
            key = " ".join(body.lower().split())
            if key in seen_bodies:
                changes.append(f'document "{name}": removed, same body as "{seen_bodies[key]}"')
                continue
            seen_bodies[key] = name
        documents.append(doc)
    return {**data, "documents": documents}, changes
