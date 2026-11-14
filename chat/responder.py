"""Response layer for the Harbor chatbot.

The Flask route only calls :func:`answer_question`.

1. ``RAG_BACKEND_URL`` forwards the question to the team's retrieval service.
2. Otherwise Harbor retrieves its own sources: National Weather Service alerts
   for a named U.S. city, or a curated guidance passage. Shelters, road
   closures, power outages, and evacuation orders are answered only when a
   source contains them. Harbor says when it does not have that source.
3. ``OPENAI_API_KEY`` lets an OpenAI-compatible model read the question in
   ordinary language, choose the relevant documents, and answer from those
   documents only. If the model is unavailable, the keyword match is shown.

Grounding (story 3): the dataset comes from ``data/knowledge_base.json`` via
``chat.knowledge``. Any reworded answer, from the model or the retrieval
service, must pass ``chat.grounding.is_supported`` against its sources, or
Harbor shows the source text instead. A retrieval-service reply with no
sources is withheld.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request

from chat.grounding import is_supported
from chat.knowledge import knowledge_base
from chat.official import fetch_active_alerts
from chat.policy import (
    SAFETY_LINE,
    SYSTEM_PROMPT,
    checked_now,
    find_place,
    clip,
    gap_catalog,
    gap_topics,
    live_alert_requested,
    normalize_for_match,
)

logger = logging.getLogger(__name__)

CLOSING = (
    "This is general guidance from Harbor's curated set, not a live incident report "
    "and not an evacuation order. " + SAFETY_LINE
)

MODEL_NOTICE = "Wording from the language model, checked against the retrieved sources below."
REPLACED_NOTICE = (
    "The reworded answer included details the sources do not state, "
    "so this shows the source text directly."
)

GUIDANCE_NOTICE = (
    "General guidance. Not a live official report. "
    "Shelter locations, road closures, power outages, and evacuation orders "
    "are included only when a retrieved source states them."
)

# The dataset lives in data/knowledge_base.json (see chat/knowledge.py). Keyword
# lists are matched against a normalized question (lowercase, hyphens treated
# as spaces). Longer phrases are worth more than single words so a specific
# question beats a passage that only shares a hazard name.


def passages() -> list[dict]:
    """The current dataset. Reloaded automatically when the file changes."""
    return knowledge_base.documents()


FALLBACK_BODY = (
    "I can check active National Weather Service alerts when you name a U.S. city and state, "
    "and I can share general safety guidance for floods, hurricanes, wildfires, and earthquakes.\n\n"
    "I do not have live shelter, road-closure, power-outage, or evacuation-order feeds. "
    "If you ask for those, I will say so instead of guessing.\n\n"
    + SAFETY_LINE
)


class ResponderError(Exception):
    """A failure worth showing to the user, with a safe message."""


def answer_question(message: str, history: list[dict] | None = None) -> dict:
    """Return a reply grounded in a retrieved source, or an explicit gap."""
    history = history or []
    backend = os.environ.get("RAG_BACKEND_URL", "").strip()
    if backend:
        return _ask_rag(backend, message, history)
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return _answer_naturally(message, history)
    return PlaceholderResponder().answer(message, history)


class PlaceholderResponder:
    """Retrieve a source and answer from that source only."""

    def answer(self, message: str, history: list[dict] | None = None) -> dict:
        del history  # The external RAG and LLM callers receive history. This path is single-turn.
        normalized = normalize_for_match(message)
        checked = checked_now()
        topics = gap_topics(normalized)
        if live_alert_requested(normalized):
            return _alert_answer(message, topics, checked)
        if topics:
            return _gap_answer(topics, checked)
        passage = _best_passage(normalized)
        if passage is None:
            return _package(
                FALLBACK_BODY,
                "unavailable",
                [_gap_source(checked, text=FALLBACK_BODY)],
                checked,
                GUIDANCE_NOTICE,
            )
        body = passage["body"].rstrip() + "\n\n" + CLOSING
        return _package(body, "guidance", [_passage_source(passage, checked)], checked, GUIDANCE_NOTICE)


def _alert_answer(message: str, topics: list[tuple[str, str, str]], checked: str) -> dict:
    place = find_place(message)
    if place is None:
        body = (
            "I can check active National Weather Service alerts only for a U.S. city and state, "
            "and this question did not include one. Name the place, for example Austin, Texas. "
            "I will not guess which alerts are in effect.\n\n" + SAFETY_LINE
        )
        if topics:
            body += "\n\n" + _gap_paragraphs(topics)
        return _package(
            body,
            "unavailable",
            [_gap_source(checked, "City and state required", text=body)],
            checked,
            GUIDANCE_NOTICE,
        )

    city, state = place
    lookup = fetch_active_alerts(city, state, timeout=_timeout_seconds())
    checked = lookup.get("retrieved_at") or checked
    if not lookup.get("ok"):
        body = lookup.get("detail") or "The alert lookup did not succeed."
        body += " I will not guess whether an alert is in effect.\n\n" + SAFETY_LINE
        return _package(
            body,
            "unavailable",
            [_gap_source(checked, "National Weather Service", text=body)],
            checked,
            "The official alert feed did not return a usable result.",
        )

    alerts = lookup.get("alerts") or []
    matched = lookup.get("matched") or f"{city}, {state}"
    if not alerts:
        body = (
            f"The National Weather Service returned no active alerts for {matched}. "
            "That is not an all-clear. Local emergency management "
            "can issue instructions that do not appear in this feed."
        )
        sources = [
            {
                "id": "nws-none",
                "title": f"National Weather Service active alerts for {matched}",
                "detail": "No active alerts in the response.",
                "text": body,
                "retrieved_at": checked,
                "url": "https://www.weather.gov/",
            }
        ]
    else:
        lines = [
            f"Active National Weather Service alerts for {matched}.",
            "",
        ]
        sources = []
        for index, alert in enumerate(alerts, start=1):
            lines.append(
                f"{index}. {alert['event']} ({alert['severity']}). "
                f"Issued {alert['sent']} by {alert['sender']}. Expires {alert['expires']}."
            )
            if alert.get("instruction"):
                lines.append(alert["instruction"])
            lines.append("")
            alert_text = (
                f"{alert['event']} ({alert['severity']}). "
                f"Issued {alert['sent']} by {alert['sender']}. Expires {alert['expires']}."
            )
            if alert.get("instruction"):
                alert_text += " " + alert["instruction"]
            sources.append(
                {
                    "id": f"nws-{index}",
                    "title": alert["headline"],
                    "detail": f"{alert['sender']} · sent {alert['sent']} · expires {alert['expires']}",
                    "text": alert_text,
                    "retrieved_at": checked,
                    "url": alert.get("url") or "",
                }
            )
        body = "\n".join(lines).rstrip()
    if topics:
        body += "\n\n" + _gap_paragraphs(topics)
        sources.append(_gap_source(checked, ", ".join(topic[1] for topic in topics), text=_gap_paragraphs(topics)))
    body += (
        "\n\nThis is only the alert feed for that point. "
        "It does not list shelters, road closures, or power outages.\n\n" + SAFETY_LINE
    )
    notice = f"National Weather Service alerts for {matched}."
    return _package(body, "official", sources, checked, notice)


def _gap_answer(topics: list[tuple[str, str, str]], checked: str) -> dict:
    labels = ", ".join(topic[1] for topic in topics)
    body = (
        f"I do not have a live source for {labels}.\n\n"
        + _gap_paragraphs(topics)
        + "\n\n"
        + SAFETY_LINE
    )
    return _package(
        body,
        "unavailable",
        [_gap_source(checked, labels, text=_gap_paragraphs(topics))],
        checked,
        f"No live source is connected for {labels}.",
    )


def _gap_paragraphs(topics: list[tuple[str, str, str]]) -> str:
    return "\n\n".join(topic[2] for topic in topics)


def _gap_source(checked: str, label: str = "No live source connected", text: str = "") -> dict:
    source = {
        "id": "gap",
        "title": label,
        "detail": "Harbor did not retrieve an official record for this request.",
        "retrieved_at": checked,
    }
    if text.strip():
        source["text"] = text.strip()
    return source


def _package(reply: str, mode: str, sources: list[dict], checked: str, notice: str) -> dict:
    return {
        "reply": reply,
        "mode": mode,
        "sources": sources,
        "retrieved_at": checked,
        "notice": notice,
    }


def _answer_naturally(message: str, history: list[dict]) -> dict:
    """Let the model choose relevant documents, then answer only from those."""
    normalized = normalize_for_match(message)
    if live_alert_requested(normalized) or gap_topics(normalized):
        return PlaceholderResponder().answer(message, history)

    draft = PlaceholderResponder().answer(message, history)
    checked = draft.get("retrieved_at") or checked_now()
    documents = _reference_documents(checked)
    try:
        raw = _chat_completion(SYSTEM_PROMPT, _question_prompt(message, history, documents))
    except (ResponderError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        logger.warning("Language model unavailable: %s", exc)
        draft = dict(draft)
        draft["notice"] = (
            "The language model was unavailable. This wording comes directly from the retrieved source."
        )
        return draft

    choice = _parse_model_choice(raw)
    if choice is None:
        if raw:
            return _use_model_wording(draft, raw)
        draft = dict(draft)
        draft["notice"] = (
            "The language model returned no wording. This text comes directly from the retrieved source."
        )
        return draft

    if choice["lookup_alerts"]:
        topics = [doc["topic"] for doc in _chosen(documents, choice["passage_ids"]) if doc["kind"] == "gap"]
        return _alert_answer(message, topics, checked)

    chosen = _chosen(documents, choice["passage_ids"])
    if not chosen:
        return draft

    gaps = [doc for doc in chosen if doc["kind"] == "gap"]
    guidance = [doc for doc in chosen if doc["kind"] == "guidance"]
    if not guidance:
        return _gap_answer([doc["topic"] for doc in gaps], checked)

    reply, used_model = _grounded_wording(choice["answer"], guidance)
    if gaps:
        reply = reply.rstrip() + "\n\n" + "\n\n".join(doc["text"] for doc in gaps)
    if "emergency services" not in reply.lower():
        reply = reply.rstrip() + "\n\n" + CLOSING
    return _package(
        reply,
        "llm",
        [doc["source"] for doc in guidance + gaps],
        checked,
        MODEL_NOTICE if used_model else REPLACED_NOTICE,
    )


def _reference_documents(checked: str) -> list[dict]:
    documents = []
    for passage in passages():
        documents.append(
            {
                "id": passage["id"],
                "kind": "guidance",
                "title": passage["title"],
                "text": passage["body"].rstrip(),
                "source": _passage_source(passage, checked),
            }
        )
    for topic_id, label, guidance in gap_catalog():
        source = _gap_source(checked, label, text=guidance)
        source["id"] = f"gap-{topic_id}"
        documents.append(
            {
                "id": f"gap-{topic_id}",
                "kind": "gap",
                "title": f"No live source for {label}",
                "text": guidance,
                "topic": (topic_id, label, guidance),
                "source": source,
            }
        )
    return documents


def _question_prompt(message: str, history: list[dict], documents: list[dict]) -> str:
    lines = [
        "The user asked a natural language question about disaster preparedness or local emergency information.",
        "Choose only the documents that contain the relevant information.",
        "Reply with one JSON object and no other text:",
        '{"passage_ids": ["id"], "lookup_alerts": false, "answer": "answer using only those documents"}',
        "Rules:",
        "- passage_ids must be ids from the documents below. Use at most 3.",
        "- Set lookup_alerts to true only when the user wants current weather alerts or warnings for a place.",
        "- If the user wants a live shelter, road closure, power outage, or evacuation order, choose the matching gap document and do not invent one.",
        "- The answer may use only facts stated in the chosen documents.",
        "- If none of the documents apply, use an empty passage_ids list and an empty answer.",
        "",
        "Documents:",
    ]
    for document in documents:
        lines.append(f"[{document['id']}] {document['title']}")
        lines.append(document["text"])
        lines.append("")
    earlier = []
    for turn in history[-6:]:
        role = turn.get("role") if isinstance(turn, dict) else None
        content = turn.get("content") if isinstance(turn, dict) else None
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            earlier.append(f"{role}: {clip(content, 400)}")
    if earlier:
        lines.append("Earlier conversation:")
        lines.extend(earlier)
        lines.append("")
    lines.append("Question: " + message.strip())
    return "\n".join(lines)


def _parse_model_choice(raw: str) -> dict | None:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if "passage_ids" not in data and "lookup_alerts" not in data:
        return None
    ids = data.get("passage_ids") or []
    if isinstance(ids, str):
        ids = [ids]
    if not isinstance(ids, list):
        ids = []
    cleaned = []
    for item in ids:
        if isinstance(item, str) and item.strip():
            cleaned.append(item.strip())
    answer = data.get("answer") if isinstance(data.get("answer"), str) else ""
    return {
        "passage_ids": cleaned[:3],
        "lookup_alerts": bool(data.get("lookup_alerts")),
        "answer": answer.strip(),
    }


def _chosen(documents: list[dict], ids: list[str]) -> list[dict]:
    by_id = {document["id"]: document for document in documents}
    chosen = []
    for document_id in ids:
        document = by_id.get(document_id)
        if document is not None and document not in chosen:
            chosen.append(document)
    return chosen


def _grounded_wording(answer: str, documents: list[dict]) -> tuple[str, bool]:
    """Keep the model's wording only if every sentence is backed by the chosen documents.

    Returns (reply, used_model_wording). When the check fails, the reply is the
    documents' own text, so nothing unsupported reaches the user.
    """
    stitched = "\n\n".join(document["text"].rstrip() for document in documents) + "\n\n" + CLOSING
    allowed = [document["text"] for document in documents] + [CLOSING]
    if is_supported(answer, allowed):
        return answer.strip(), True
    if answer.strip():
        logger.info("Model wording rejected: not supported by %s", [d["id"] for d in documents])
    return stitched, False


def _use_model_wording(draft: dict, phrased: str) -> dict:
    """Use free-text model wording only if the draft's sources back up every sentence."""
    source_texts = [source.get("text", "") for source in draft.get("sources") or []]
    if draft.get("mode") != "guidance" or not is_supported(phrased, source_texts + [CLOSING]):
        kept = dict(draft)
        kept["notice"] = REPLACED_NOTICE
        return kept
    phrased_draft = dict(draft)
    phrased_draft["reply"] = phrased
    phrased_draft["mode"] = "llm"
    phrased_draft["notice"] = MODEL_NOTICE
    return phrased_draft


def _chat_completion(system: str, user: str) -> str:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    payload = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '').strip()}",
            "User-Agent": "Harbor/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_timeout_seconds()) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise ResponderError("The language model returned an error.") from exc
    data = json.loads(raw)
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        return ""
    content = ((choices[0] or {}).get("message") or {}).get("content")
    if not isinstance(content, str):
        return ""
    return content.strip()


def _ask_rag(url: str, message: str, history: list[dict]) -> dict:
    payload = json.dumps({"message": message, "history": history}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Harbor/1.0",
        },
        method="POST",
    )
    timeout = _timeout_seconds()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        logger.warning("RAG backend returned HTTP %s", exc.code)
        raise ResponderError("The retrieval service returned an error. Please try again.") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("RAG backend unreachable: %s", exc)
        raise ResponderError("The retrieval service is unavailable. Please try again.") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ResponderError("The retrieval service sent an unreadable response.") from exc
    if not isinstance(data, dict):
        raise ResponderError("The retrieval service sent an unreadable response.")

    reply = data.get("reply") or data.get("answer") or data.get("response")
    if not isinstance(reply, str) or not reply.strip():
        raise ResponderError("The retrieval service did not return an answer.")

    return _check_rag_reply(reply.strip(), _normalize_sources(data.get("sources")))


def _check_rag_reply(reply: str, sources: list[dict]) -> dict:
    """Show a retrieval-service answer only when it is tied to its sources."""
    checked = checked_now()
    if not sources:
        logger.warning("RAG backend answered without sources; reply withheld")
        body = (
            "The retrieval service answered without naming any source, so Harbor is not showing "
            "that answer. Please try rephrasing the question.\n\n" + SAFETY_LINE
        )
        return _package(
            body,
            "unavailable",
            [_gap_source(checked, "No source returned", text=body)],
            checked,
            "Answer withheld: no source was returned.",
        )
    texts = [source.get("text", "") for source in sources]
    if not any(text.strip() for text in texts):
        return _package(
            reply,
            "rag",
            sources,
            checked,
            "The retrieval service named its sources but did not send their text, "
            "so Harbor could not check this answer against them.",
        )
    if is_supported(reply, texts + [SAFETY_LINE, CLOSING]):
        return _package(reply, "rag", sources, checked, None)
    logger.warning("RAG backend reply not supported by its sources; showing source text instead")
    stitched = "\n\n".join(
        f"{source['title']}: {source['text']}" for source in sources if source.get("text", "").strip()
    )
    return _package(stitched + "\n\n" + SAFETY_LINE, "rag", sources, checked, REPLACED_NOTICE)


def _timeout_seconds() -> float:
    raw = os.environ.get("RAG_TIMEOUT", "30")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 30.0
    return min(max(value, 1.0), 120.0)


def _normalize_sources(sources) -> list[dict]:
    if not isinstance(sources, list):
        return []
    cleaned = []
    for item in sources[:6]:
        if isinstance(item, str) and item.strip():
            cleaned.append({"title": item.strip()})
        elif isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or "").strip()
            if not title:
                continue
            entry = {"title": title}
            if item.get("id"):
                entry["id"] = str(item["id"])[:80]
            document = item.get("text") if isinstance(item.get("text"), str) else ""
            if not document.strip() and isinstance(item.get("snippet"), str):
                document = item.get("snippet")
            detail = item.get("detail")
            if not (isinstance(detail, str) and detail.strip()) and document.strip() and document is not item.get("text"):
                detail = document
            if isinstance(detail, str) and detail.strip():
                entry["detail"] = detail.strip()[:300]
            if document.strip():
                entry["text"] = document.strip()[:4000]
            retrieved_at = item.get("retrieved_at")
            if isinstance(retrieved_at, str) and retrieved_at.strip():
                entry["retrieved_at"] = retrieved_at.strip()[:40]
            url = item.get("url")
            if isinstance(url, str) and (url.startswith("https://") or url.startswith("http://")):
                entry["url"] = url
            cleaned.append(entry)
    return cleaned


def _best_passage(question: str) -> dict | None:
    normalized = _normalize(question)
    best = None
    best_score = 0
    current = passages()
    for passage in current:
        if passage.get("broad"):
            continue
        score = _score(normalized, passage)
        if score > best_score:
            best = passage
            best_score = score
    if best is not None:
        return best
    for passage in current:
        if passage.get("broad") and _score(normalized, passage) > 0:
            return passage
    return None


def _passage_source(passage: dict, checked: str) -> dict:
    """How a dataset document is shown to the user as a source."""
    detail = passage.get("source") or "Harbor curated guidance"
    if passage.get("updated"):
        detail += f" · updated {passage['updated']}"
    detail += " · Not a live official report."
    source = {
        "id": passage["id"],
        "title": passage["title"],
        "detail": detail,
        "text": passage["body"].rstrip(),
        "retrieved_at": checked,
    }
    if passage.get("url"):
        source["url"] = passage["url"]
    return source


def _score(normalized_question: str, passage: dict) -> int:
    score = 0
    for keyword in passage["keywords"]:
        if keyword in normalized_question:
            score += len(keyword.split()) + 1
    return score


def _normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("'", "")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
