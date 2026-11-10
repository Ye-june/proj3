"""Live retrieval from official public feeds.

National Weather Service alerts are the only live operational feed Harbor
calls directly. A failed or empty lookup is reported as such.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from chat.policy import checked_now, clip, state_name

logger = logging.getLogger(__name__)

USER_AGENT = "Harbor/1.0 (disaster-information assistant; educational)"


def fetch_active_alerts(city: str, state: str, timeout: float = 8.0) -> dict:
    """Look up active NWS alerts for a city. Never fills in alerts that were not returned."""
    retrieved_at = checked_now()
    try:
        match = _geocode(city, state, timeout)
    except _FeedError as exc:
        logger.warning("Location lookup failed: %s", exc)
        return {
            "ok": False,
            "reason": "unavailable",
            "matched": "",
            "alerts": [],
            "retrieved_at": retrieved_at,
            "detail": "The location lookup did not respond, so no National Weather Service request was sent.",
        }
    if match is None:
        return {
            "ok": False,
            "reason": "no_match",
            "matched": "",
            "alerts": [],
            "retrieved_at": retrieved_at,
            "detail": f"No U.S. location match was found for {city}, {state}.",
        }
    try:
        alerts = _alerts_at(match["lat"], match["lon"], timeout)
    except _FeedError as exc:
        logger.warning("NWS alerts failed: %s", exc)
        return {
            "ok": False,
            "reason": "unavailable",
            "matched": match["label"],
            "alerts": [],
            "retrieved_at": retrieved_at,
            "detail": "The National Weather Service alert feed did not return a usable response.",
        }
    return {
        "ok": True,
        "reason": None,
        "matched": match["label"],
        "alerts": alerts,
        "retrieved_at": retrieved_at,
        "detail": None,
    }


class _FeedError(Exception):
    pass


def _geocode(city: str, state: str, timeout: float) -> dict | None:
    """Resolve a city to coordinates. Alert text still comes from NWS, not this lookup."""
    query = urllib.parse.urlencode(
        {"name": city, "count": "10", "countryCode": "US", "format": "json"}
    )
    payload = _get_json(
        f"https://geocoding-api.open-meteo.com/v1/search?{query}",
        {"Accept": "application/json"},
        timeout,
    )
    wanted = state_name(state).lower()
    for item in payload.get("results") or []:
        admin = str(item.get("admin1") or "").lower()
        if admin != wanted:
            continue
        try:
            lat = float(item["latitude"])
            lon = float(item["longitude"])
        except (KeyError, TypeError, ValueError) as exc:
            raise _FeedError("coordinates missing") from exc
        label = f"{item.get('name') or city}, {item.get('admin1') or state}"
        return {"lat": lat, "lon": lon, "label": label}
    return None


def _alerts_at(lat: float, lon: float, timeout: float) -> list[dict]:
    point = f"{lat:.4f},{lon:.4f}"
    payload = _get_json(
        f"https://api.weather.gov/alerts/active?point={point}",
        {"Accept": "application/geo+json"},
        timeout,
    )
    alerts = []
    for feature in (payload.get("features") or [])[:3]:
        props = feature.get("properties") or {}
        event = str(props.get("event") or "Weather alert").strip()
        headline = str(props.get("headline") or event).strip()
        instruction = clip(str(props.get("instruction") or props.get("description") or ""))
        alerts.append(
            {
                "event": event,
                "headline": headline,
                "severity": str(props.get("severity") or "not stated"),
                "sender": str(props.get("senderName") or "National Weather Service"),
                "sent": _short_time(props.get("sent") or props.get("effective")),
                "expires": _short_time(props.get("expires") or props.get("ends")),
                "instruction": instruction,
                "url": _http_url(props.get("id") or feature.get("id")),
            }
        )
    return alerts


def _get_json(url: str, accept: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, **accept},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise _FeedError(f"HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise _FeedError(str(exc)) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _FeedError("unreadable json") from exc
    if not isinstance(payload, dict):
        raise _FeedError("unexpected payload")
    return payload


def _short_time(value) -> str:
    if not isinstance(value, str) or not value.strip():
        return "time not provided"
    return value.strip().replace("T", " ")[:16]


def _http_url(value) -> str:
    if not isinstance(value, str):
        return ""
    if value.startswith("https://") or value.startswith("http://"):
        return value
    return ""
