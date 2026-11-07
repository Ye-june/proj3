"""Response layer for the Harbor chatbot.

PLACEHOLDER: until the team's retrieval pipeline is connected, questions are
answered from the curated preview set in ``PASSAGES``. The Flask route only
calls :func:`answer_question`, so the interface does not need to change later.

Connect retrieval in either of these ways:

1. Set ``RAG_BACKEND_URL``. Harbor POSTs
   ``{"message": str, "history": [{"role", "content"}]}`` and expects
   ``{"reply": str, "sources": [str | {"title": str}]}``.
2. Replace the body of :func:`answer_question` with a direct import of the
   team's RAG module, keeping the same return shape.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

CLOSING = (
    "This is general planning guidance, not an evacuation order. "
    "If you are in danger, follow your local emergency officials."
)

PREVIEW_NOTICE = (
    "Preview knowledge set. Answers are curated starter guidance until "
    "retrieval is connected."
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
    "Harbor can help you plan ahead for floods, hurricanes, wildfires, and earthquakes. "
    "The questions it is built for are the ones households keep having to answer separately "
    "for each hazard.\n\n"
    "You can ask what belongs in one go-bag, which alerts to join before the season, how "
    "leaving for a flood differs from leaving for a hurricane, what to do while the ground "
    "is shaking, when to leave for a wildfire, or how to pick a meeting place if you are separated.\n\n"
    "Try one of those. If you are in danger right now, use local emergency services and official "
    "instructions. This chat is for planning, not for a live incident."
)


class ResponderError(Exception):
    """A failure worth showing to the user, with a safe message."""


def answer_question(message: str, history: list[dict] | None = None) -> dict:
    """Return a reply dict. Uses the RAG service when configured, else the preview set."""
    history = history or []
    backend = os.environ.get("RAG_BACKEND_URL", "").strip()
    if backend:
        return _ask_rag(backend, message, history)
    return PlaceholderResponder().answer(message, history)


class PlaceholderResponder:
    """PLACEHOLDER. Curated preview answers, clearly labeled in the response."""

    def answer(self, message: str, history: list[dict] | None = None) -> dict:
        del history  # Forwarded for the future RAG call. The preview set is single-turn.
        passage = _best_passage(message)
        if passage is None:
            body = FALLBACK_BODY
            sources = [{"id": "preview", "title": "Harbor preview set", "detail": PREVIEW_NOTICE}]
        else:
            body = passage["body"].rstrip() + "\n\n" + CLOSING
            sources = [
                {
                    "id": passage["id"],
                    "title": passage["title"],
                    "detail": "Preview knowledge set",
                }
            ]
        return {
            "reply": body,
            "mode": "placeholder",
            "sources": sources,
            "notice": PREVIEW_NOTICE,
        }


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
