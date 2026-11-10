"""Response layer for the Harbor chatbot.

The Flask route only calls :func:`answer_question`.

1. ``RAG_BACKEND_URL`` forwards the question to the team's retrieval service.
2. Otherwise Harbor retrieves its own sources: National Weather Service alerts
   for a named U.S. city, or a curated guidance passage. Shelters, road
   closures, power outages, and evacuation orders are answered only when a
   source contains them. Harbor says when it does not have that source.
3. ``OPENAI_API_KEY`` optionally rephrases the retrieved text through an
   OpenAI-compatible Chat Completions API. The model is instructed to add
   no facts beyond that text.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request

from chat.official import fetch_active_alerts
from chat.policy import (
    SAFETY_LINE,
    SYSTEM_PROMPT,
    checked_now,
    find_place,
    gap_topics,
    live_alert_requested,
    normalize_for_match,
)

logger = logging.getLogger(__name__)

CLOSING = (
    "This is general guidance from Harbor's curated set, not a live incident report "
    "and not an evacuation order. " + SAFETY_LINE
)

GUIDANCE_NOTICE = (
    "General guidance. Not a live official report. "
    "Shelter locations, road closures, power outages, and evacuation orders "
    "are included only when a retrieved source states them."
)

# Keyword lists are matched against a normalized question (lowercase, hyphens
# treated as spaces). Longer phrases are worth more than single words so a
# specific question beats a passage that only shares a hazard name.
PASSAGES = [
    {
        "id": "gobag",
        "title": "One go-bag for several hazards",
        "keywords": [
            "go bag",
            "emergency kit",
            "what should go in",
            "what to pack",
            "gallon",
            "supplies",
            "power bank",
            "prescription",
            "pet supplies",
        ],
        "body": (
            "Pack one bag the household can carry or load quickly. The same bag is the right start "
            "for a flood, a hurricane, a wildfire, or an earthquake, because the first problem is "
            "often the same: you may have to leave, or home may be without power and open stores.\n\n"
            "For each person, a widely used planning baseline is one gallon of water per day for at "
            "least three days, plus food that does not need a stove. Add a flashlight, batteries, a "
            "whistle, a phone power bank, a small first-aid kit, and the prescription medicines people "
            "in the household actually take. Include copies of identification and insurance, a little "
            "cash in small bills, sturdy shoes, and a change of clothes.\n\n"
            "If a child, an older adult, or a pet is part of the household, pack what they need "
            "specifically. Glasses, hearing aids, and a paper list of phone numbers belong in the bag, "
            "because the cell network may not work. Keep the bag by the door you would use to leave. "
            "During the months when local hazards are more active, keep the car at least half full."
        ),
    },
    {
        "id": "compare",
        "title": "Flood decisions and hurricane decisions",
        "keywords": [
            "leaving for a flood",
            "leaving for a hurricane",
            "different from leaving",
            "storm surge",
            "evacuation zone",
            "floodwater",
            "flash flood",
            "drive through",
            "turn around",
            "do not drive",
        ],
        "body": (
            "A flood and a hurricane can both require leaving, and both can involve water. The decision "
            "is still not the same.\n\n"
            "A hurricane often gives a day or more of warning. The dangerous mistake is waiting to see "
            "the wind. If your zone is ordered to evacuate, go while roads are still open. Storm surge "
            "can flood coastal and tidal areas well before the center of the storm arrives, including "
            "places that do not look like a beach. Learn your evacuation zone on a clear day.\n\n"
            "A flood can be a river that rises for hours, or a flash flood that arrives in minutes. "
            "Leave if officials advise it. Separately, do not drive or walk through floodwater, even "
            "on a route you were not using to evacuate. The road underneath may be gone, and the water "
            "is often deeper and faster than it looks. If you meet standing water, turn around.\n\n"
            "Shared preparation is a go-bag, enough fuel to leave, two ways out of the neighborhood, "
            "and a meeting place. The split is this: a hurricane zone order is a reason to leave early. "
            "Floodwater on the road is a reason to stop, not to push through."
        ),
    },
    {
        "id": "alerts",
        "title": "Alerts to join before the season",
        "keywords": [
            "which alerts",
            "alerts should",
            "join before",
            "hurricane season",
            "wildfire season",
            "wireless emergency",
            "weather radio",
            "alert list",
            "sign up",
            "notifications",
        ],
        "body": (
            "Join alerts before you need them. Wireless Emergency Alerts are the short official messages "
            "a phone can receive for imminent threats. Leave them on. They do not replace a local list.\n\n"
            "Also join the system run by your city, county, or parish emergency management office. That "
            "list is how many communities send evacuation notices, shelter openings, and boil-water or "
            "air-quality messages that a national push will not name street by street. A NOAA Weather "
            "Radio is worth having where hurricanes or floods are part of the local risk, because it "
            "still works when the internet does not.\n\n"
            "Before wildfire season, confirm that the county list is the one that will name your area, "
            "and choose an out-of-area contact. Before hurricane season, confirm whether the home is in "
            "an evacuation zone and save the official page you will check. Alert systems are a poor "
            "thing to learn while you are packing."
        ),
    },
    {
        "id": "earthquake",
        "title": "What to do while the ground is shaking",
        "keywords": [
            "during an earthquake",
            "run outside",
            "stay where we are",
            "drop cover",
            "hold on",
            "aftershock",
            "while it is shaking",
            "doorway",
        ],
        "body": (
            "While the ground is shaking, Drop, Cover, and Hold On. Get down, get under a sturdy table "
            "or against an interior wall away from glass, and hold on until the shaking stops. Do not "
            "run outside during the shaking. Falling brick, glass, and pieces of the building front are "
            "a common way people are hurt. A doorway is not a special safe spot.\n\n"
            "When the shaking stops, check yourself and the people with you. If you smell gas, or the "
            "building is badly damaged, leave then and move to an open area away from walls and wires. "
            "Expect aftershocks. Do not use elevators.\n\n"
            "This is the opposite of a wildfire evacuation or a hurricane zone order, where the safer "
            "move is often to leave before conditions get worse. During shaking, stay and cover. After "
            "shaking, leave only if the building is unsafe or officials say so."
        ),
    },
    {
        "id": "wildfire",
        "title": "Wildfire: ready, set, go",
        "keywords": [
            "ready set go",
            "wildfire",
            "smoke",
            "n95",
            "see flames",
            "defensible",
            "fire weather",
            "air quality",
        ],
        "body": (
            "Treat wildfire risk as Ready, Set, Go.\n\n"
            "Ready, before a fire is anywhere near you: the go-bag is packed, copies of documents are "
            "with it, you know the route out, and brush and debris immediately around the home have been "
            "reduced where that is your responsibility. Set, when fire weather is high: the bag is at "
            "the door, the car has fuel, you have N95 respirators for smoke, and you are watching the "
            "local emergency channel. Go means leave early. Waiting until you can see flames often means "
            "driving in smoke, in traffic, on the only road out.\n\n"
            "If you have not been told to leave, staying inside with windows and doors closed is safer "
            "than standing in heavy smoke. An N95 helps for a short trip through smoky air. It is not a "
            "reason to ignore an evacuation notice. If you are ordered to leave, leave."
        ),
    },
    {
        "id": "plan",
        "title": "A meeting plan that works for every hazard",
        "keywords": [
            "meeting place",
            "out of area",
            "get separated",
            "find each other",
            "reunion",
            "family plan",
            "contact person",
            "if we are separated",
        ],
        "body": (
            "A plan that works for every hazard is mostly about finding each other.\n\n"
            "Pick two meeting places and say them out loud. One should be just outside the home, in case "
            "you step out after earthquake shaking or because an alarm is sounding. One should be outside "
            "the neighborhood, in case you cannot go back. Choose one out-of-area contact, someone far "
            "enough away that the same storm or fire is unlikely to reach them, and agree that everyone "
            "will report to that person. Local phone lines get congested. A distant contact often gets "
            "through when a neighbor does not.\n\n"
            "Write the places and the number on paper, and put a copy in the go-bag. Photograph "
            "identification, insurance, and prescriptions and keep those copies with the list. Tell "
            "children, roommates, and anyone who is often in the home. You do not need a different "
            "reunion plan for a flood, a hurricane, a wildfire, and an earthquake."
        ),
    },
    {
        "id": "overview",
        "title": "What to prepare once",
        "keywords": [
            "where do i start",
            "how do i prepare",
            "how should i prepare",
            "how should we prepare",
            "what should i do first",
            "what should we do first",
            "getting started",
            "what stays the same",
            "what changes",
            "across hazards",
            "before the next",
        ],
        "body": (
            "A useful plan covers the actions that transfer, then the few decisions that do not.\n\n"
            "Do these once for floods, hurricanes, wildfires, and earthquakes. Pack one go-bag and keep "
            "it reachable. Leave phone emergency alerts on, and join the local alert list before the "
            "season rather than during it. Choose two meeting places and one out-of-area contact. Keep "
            "copies of identification, insurance, and a medication list with the bag.\n\n"
            "Then learn the split. During earthquake shaking: Drop, Cover, and Hold On, and do not run "
            "outside while it is shaking. In a flood: do not walk or drive through water, and leave "
            "before routes are cut off if officials advise it. For a hurricane: if your zone is ordered "
            "out, leave before wind and surge arrive. For a wildfire: leave early enough that you are "
            "not driving through smoke and traffic.\n\n"
            "Ask Harbor to go deeper on any one of those. Start with the shared steps, then the decision "
            "that belongs to the hazard you actually face."
        ),
    },
]

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
    draft = PlaceholderResponder().answer(message, history)
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return _phrase_with_llm(message, draft)
    return draft


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
            return _package(FALLBACK_BODY, "unavailable", [_gap_source(checked)], checked, GUIDANCE_NOTICE)
        body = passage["body"].rstrip() + "\n\n" + CLOSING
        source = {
            "id": passage["id"],
            "title": passage["title"],
            "detail": "Curated guidance. Not a live official report.",
            "retrieved_at": checked,
        }
        return _package(body, "guidance", [source], checked, GUIDANCE_NOTICE)


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
        return _package(body, "unavailable", [_gap_source(checked, "City and state required")], checked, GUIDANCE_NOTICE)

    city, state = place
    lookup = fetch_active_alerts(city, state, timeout=_timeout_seconds())
    checked = lookup.get("retrieved_at") or checked
    if not lookup.get("ok"):
        body = lookup.get("detail") or "The alert lookup did not succeed."
        body += " I will not guess whether an alert is in effect.\n\n" + SAFETY_LINE
        return _package(
            body,
            "unavailable",
            [_gap_source(checked, "National Weather Service")],
            checked,
            "The official alert feed did not return a usable result.",
        )

    alerts = lookup.get("alerts") or []
    matched = lookup.get("matched") or f"{city}, {state}"
    if not alerts:
        body = (
            f"The National Weather Service returned no active alerts for {matched} "
            f"when checked at {checked}. That is not an all-clear. Local emergency management "
            "can issue instructions that do not appear in this feed."
        )
        sources = [
            {
                "id": "nws-none",
                "title": f"National Weather Service active alerts for {matched}",
                "detail": "No active alerts in the response.",
                "retrieved_at": checked,
                "url": "https://www.weather.gov/",
            }
        ]
    else:
        lines = [
            f"Active National Weather Service alerts for {matched}, checked at {checked}.",
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
            sources.append(
                {
                    "id": f"nws-{index}",
                    "title": alert["headline"],
                    "detail": f"{alert['sender']} · sent {alert['sent']} · expires {alert['expires']}",
                    "retrieved_at": checked,
                    "url": alert.get("url") or "",
                }
            )
        body = "\n".join(lines).rstrip()
    if topics:
        body += "\n\n" + _gap_paragraphs(topics)
    body += (
        "\n\nThis is only the alert feed for that point. "
        "It does not list shelters, road closures, or power outages.\n\n" + SAFETY_LINE
    )
    notice = f"National Weather Service alerts for {matched}, checked at {checked}."
    return _package(body, "official", sources, checked, notice)


def _gap_answer(topics: list[tuple[str, str, str]], checked: str) -> dict:
    labels = ", ".join(topic[1] for topic in topics)
    body = (
        f"I do not have a live source for {labels}, checked at {checked}.\n\n"
        + _gap_paragraphs(topics)
        + "\n\n"
        + SAFETY_LINE
    )
    return _package(
        body,
        "unavailable",
        [_gap_source(checked, labels)],
        checked,
        f"No live source is connected for {labels}.",
    )


def _gap_paragraphs(topics: list[tuple[str, str, str]]) -> str:
    return "\n\n".join(topic[2] for topic in topics)


def _gap_source(checked: str, label: str = "No live source connected") -> dict:
    return {
        "id": "gap",
        "title": label,
        "detail": "Harbor did not retrieve an official record for this request.",
        "retrieved_at": checked,
    }


def _package(reply: str, mode: str, sources: list[dict], checked: str, notice: str) -> dict:
    return {
        "reply": reply,
        "mode": mode,
        "sources": sources,
        "retrieved_at": checked,
        "notice": notice,
    }


def _phrase_with_llm(message: str, draft: dict) -> dict:
    """Rephrase retrieved text. On failure, return the source text unchanged."""
    context = (
        f"Question: {message}\n\n"
        f"Retrieved at: {draft.get('retrieved_at', '')}\n"
        f"Mode: {draft.get('mode', '')}\n"
        f"Source text:\n{draft.get('reply', '')}\n"
    )
    try:
        phrased = _chat_completion(SYSTEM_PROMPT, context)
    except (ResponderError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        logger.warning("Language model unavailable: %s", exc)
        draft = dict(draft)
        draft["notice"] = (
            "The language model was unavailable. This wording comes directly from the retrieved source."
        )
        return draft
    if not phrased:
        draft = dict(draft)
        draft["notice"] = (
            "The language model returned no wording. This text comes directly from the retrieved source."
        )
        return draft
    phrased_draft = dict(draft)
    phrased_draft["reply"] = phrased
    phrased_draft["mode"] = "llm"
    phrased_draft["notice"] = "Wording from the language model, limited to the retrieved sources below."
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

    return {
        "reply": reply.strip(),
        "mode": "rag",
        "sources": _normalize_sources(data.get("sources")),
        "notice": None,
    }


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
            detail = item.get("detail") or item.get("snippet")
            if isinstance(detail, str) and detail.strip():
                entry["detail"] = detail.strip()[:300]
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
    for passage in PASSAGES:
        if passage["id"] == "overview":
            continue
        score = _score(normalized, passage)
        if score > best_score:
            best = passage
            best_score = score
    if best is not None:
        return best
    overview = next(passage for passage in PASSAGES if passage["id"] == "overview")
    if _score(normalized, overview) > 0:
        return overview
    return None


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
