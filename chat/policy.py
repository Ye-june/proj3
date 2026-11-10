"""Safety policy for Harbor.

The chatbot may describe only what a retrieved source actually says. Live
shelter lists, road closures, power outages, and evacuation orders are not
invented when no source returned them.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

SYSTEM_PROMPT = """You are an AI-powered disaster response chatbot accessible through a hosted web application. Your purpose is to help individuals affected by natural disasters, as well as emergency response teams, quickly access accurate, reliable, and up-to-date information.

Using Retrieval-Augmented Generation (RAG), retrieve information from trusted sources such as official emergency alerts, government announcements, emergency services, and infrastructure reports. Provide clear, concise, and context-specific answers tailored to each user's needs, including evacuation guidance, shelter locations, road closures, power outages, and available emergency resources.

Prioritize information accuracy, timeliness, and user safety. Include verifiable sources and timestamps when available, clearly communicate uncertainty, and never invent emergency information. If information is unavailable or outdated, explain this transparently and direct users to official emergency services.

Your goal is to reduce confusion, improve access to critical information, and support informed decision-making during natural disasters, especially when communication and emergency response systems are disrupted. The chatbot should complement, not replace, official emergency services.

Additional rules for this request:
- Use only the retrieved source text in the user message. Do not add shelters, routes, closures, outages, orders, or alert details that are not written there.
- Repeat the source name and the time it was checked when you state a fact.
- If the source is general guidance, say it is not a live incident report.
- If the person may be in immediate danger, tell them to contact local emergency services.
- Rephrase for clarity. Do not broaden the claim.
"""

_STATES = {
    "al": "AL", "alabama": "AL",
    "ak": "AK", "alaska": "AK",
    "az": "AZ", "arizona": "AZ",
    "ar": "AR", "arkansas": "AR",
    "ca": "CA", "california": "CA",
    "co": "CO", "colorado": "CO",
    "ct": "CT", "connecticut": "CT",
    "de": "DE", "delaware": "DE",
    "dc": "DC", "district of columbia": "DC",
    "fl": "FL", "florida": "FL",
    "ga": "GA", "georgia": "GA",
    "hi": "HI", "hawaii": "HI",
    "id": "ID", "idaho": "ID",
    "il": "IL", "illinois": "IL",
    "in": "IN", "indiana": "IN",
    "ia": "IA", "iowa": "IA",
    "ks": "KS", "kansas": "KS",
    "ky": "KY", "kentucky": "KY",
    "la": "LA", "louisiana": "LA",
    "me": "ME", "maine": "ME",
    "md": "MD", "maryland": "MD",
    "ma": "MA", "massachusetts": "MA",
    "mi": "MI", "michigan": "MI",
    "mn": "MN", "minnesota": "MN",
    "ms": "MS", "mississippi": "MS",
    "mo": "MO", "missouri": "MO",
    "mt": "MT", "montana": "MT",
    "ne": "NE", "nebraska": "NE",
    "nv": "NV", "nevada": "NV",
    "nh": "NH", "new hampshire": "NH",
    "nj": "NJ", "new jersey": "NJ",
    "nm": "NM", "new mexico": "NM",
    "ny": "NY", "new york": "NY",
    "nc": "NC", "north carolina": "NC",
    "nd": "ND", "north dakota": "ND",
    "oh": "OH", "ohio": "OH",
    "ok": "OK", "oklahoma": "OK",
    "or": "OR", "oregon": "OR",
    "pa": "PA", "pennsylvania": "PA",
    "ri": "RI", "rhode island": "RI",
    "sc": "SC", "south carolina": "SC",
    "sd": "SD", "south dakota": "SD",
    "tn": "TN", "tennessee": "TN",
    "tx": "TX", "texas": "TX",
    "ut": "UT", "utah": "UT",
    "vt": "VT", "vermont": "VT",
    "va": "VA", "virginia": "VA",
    "wa": "WA", "washington": "WA",
    "wv": "WV", "west virginia": "WV",
    "wi": "WI", "wisconsin": "WI",
    "wy": "WY", "wyoming": "WY",
}

_GAP_TOPICS = (
    (
        "shelter",
        (
            "shelter",
            "warming center",
            "cooling center",
            "evacuation center",
            "place to sleep",
            "somewhere to stay",
        ),
        "open shelters",
        "I do not have a live list of open shelters, and I will not invent one. "
        "Check your local emergency management office or the American Red Cross shelter map, "
        "or call 211. If you are in danger, call 911.",
    ),
    (
        "roads",
        (
            "road closure",
            "roads closed",
            "road closed",
            "highway closed",
            "which roads",
            "is the highway open",
            "which route is open",
        ),
        "road closures",
        "I do not have a live road-closure feed, and I will not guess which roads are open. "
        "Use your state transportation department or local emergency management for closures.",
    ),
    (
        "power",
        (
            "power outage",
            "power is out",
            "without power",
            "blackout",
            "electricity is out",
            "lights are out",
            "lost power",
        ),
        "power outages",
        "I do not have a live outage map, and I will not guess where power is out. "
        "Check the utility that serves the address, or the outage page linked by local emergency management.",
    ),
    (
        "order",
        (
            "evacuation order",
            "ordered to evacuate",
            "under an order",
            "am i supposed to leave",
            "do we have to evacuate",
            "mandatory evacuation",
        ),
        "evacuation orders",
        "I cannot see an evacuation order for you, and I will not invent one. "
        "An order comes from local emergency management. This chat is not that order.",
    ),
)

_SIGNUP_PHRASES = ("sign up", "join before", "which alerts should", "alert list", "alerts should we join")
_LIVE_ALERT_PHRASES = (
    "active alert",
    "active weather",
    "weather alert",
    "weather alerts",
    "in effect",
    "current alert",
    "national weather service",
    " nws",
)

SAFETY_LINE = (
    "If you are in danger, contact local emergency services. "
    "Harbor complements those services and does not replace them."
)


def state_name(code: str) -> str:
    for name, abbreviation in _STATES.items():
        if abbreviation == code and len(name) > 2:
            return name
    return code


def checked_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def normalize_for_match(text: str) -> str:
    text = text.lower().replace("’", "'").replace("'", "")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_place(message: str) -> tuple[str, str] | None:
    """Return a city and postal state when the question names one."""
    text = message.lower().replace("’", "'")
    pattern = re.compile(r"(?:in|near|around|for|at)\s+([a-z][a-z .'-]{1,40}?)\s*,\s*")
    found = None
    for match in pattern.finditer(text):
        city = re.sub(r"\s+", " ", match.group(1)).strip(" .")
        words = re.findall(r"[a-z]+", text[match.end() :])[:3]
        state = None
        if len(words) >= 2:
            state = _STATES.get(f"{words[0]} {words[1]}")
        if state is None and words:
            state = _STATES.get(words[0])
        if city and state:
            found = (city.title(), state)
    return found


def live_alert_requested(normalized: str) -> bool:
    padded = f" {normalized} "
    if any(phrase in normalized for phrase in _SIGNUP_PHRASES):
        return False
    return any(phrase in padded or phrase in normalized for phrase in _LIVE_ALERT_PHRASES)


def gap_topics(normalized: str) -> list[tuple[str, str, str]]:
    found = []
    for topic_id, phrases, label, guidance in _GAP_TOPICS:
        if any(phrase in normalized for phrase in phrases):
            found.append((topic_id, label, guidance))
    return found


def gap_catalog() -> list[tuple[str, str, str]]:
    """Every live-data gap, so a question can be matched without a fixed phrase."""
    return [(topic_id, label, guidance) for topic_id, phrases, label, guidance in _GAP_TOPICS]


def clip(text: str, limit: int = 420) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
