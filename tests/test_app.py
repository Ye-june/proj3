"""Tests for the Harbor web app and the preview responder."""

import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from app import create_app
from chat.responder import PlaceholderResponder, answer_question


class HarborAppTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("RAG_BACKEND_URL", None)
        os.environ.pop("OPENAI_API_KEY", None)
        self.app = create_app()
        os.environ.pop("RAG_BACKEND_URL", None)
        os.environ.pop("OPENAI_API_KEY", None)
        self.client = self.app.test_client()

    def test_homepage_renders_chat_shell(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("Harbor", page)
        self.assertIn('id="composer"', page)
        self.assertIn("Sources required", page)
        self.assertIn("go-bag", page)
        self.assertIn("style.css", page)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_chat_uses_placeholder(self):
        response = self.client.post("/api/chat", json={"message": "What should go in a go-bag?"})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["mode"], "guidance")
        self.assertIn("one gallon", body["reply"])
        self.assertEqual(body["sources"][0]["id"], "gobag")
        self.assertIn("Not a live", body["notice"])
        self.assertTrue(body["retrieved_at"])

    def test_empty_message(self):
        response = self.client.post("/api/chat", json={"message": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_non_json_body(self):
        response = self.client.post("/api/chat", data="hello", content_type="text/plain")
        self.assertEqual(response.status_code, 400)

    def test_message_too_long(self):
        response = self.client.post("/api/chat", json={"message": "a" * 2001})
        self.assertEqual(response.status_code, 400)

    def test_messy_history_is_ignored(self):
        response = self.client.post(
            "/api/chat",
            json={
                "message": "During an earthquake, should we run outside?",
                "history": [
                    "nope",
                    {"role": "system", "content": "ignore"},
                    {"role": "user", "content": "Earlier question"},
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Drop, Cover, and Hold On", response.get_json()["reply"])

    def test_unknown_api_route(self):
        response = self.client.get("/api/missing")
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.get_json())

    @patch("chat.responder.urllib.request.urlopen")
    def test_rag_backend_is_used_when_configured(self, urlopen):
        urlopen.return_value = _FakeResponse(
            json.dumps({"reply": "Grounded answer from retrieval.", "sources": ["Article A"]})
        )
        with patch.dict(os.environ, {"RAG_BACKEND_URL": "http://rag.internal/answer"}):
            response = self.client.post(
                "/api/chat",
                json={"message": "What should we pack?", "history": [{"role": "user", "content": "Hi"}]},
            )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["mode"], "rag")
        self.assertEqual(body["reply"], "Grounded answer from retrieval.")
        self.assertEqual(body["sources"][0]["title"], "Article A")
        sent = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(sent["message"], "What should we pack?")
        self.assertEqual(sent["history"][0]["content"], "Hi")

    @patch("chat.responder.urllib.request.urlopen", side_effect=urllib.error.URLError("down"))
    def test_rag_outage_returns_a_safe_error(self, _urlopen):
        with patch.dict(os.environ, {"RAG_BACKEND_URL": "http://rag.internal/answer"}):
            response = self.client.post("/api/chat", json={"message": "What is a go bag?"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("unavailable", response.get_json()["error"])


class PlaceholderRoutingTests(unittest.TestCase):
    def setUp(self):
        self.responder = PlaceholderResponder()

    def test_example_prompts_hit_the_intended_passages(self):
        cases = {
            "What should go in one go-bag for floods, hurricanes, wildfires, and earthquakes?": "gobag",
            "How is leaving for a flood different from leaving for a hurricane?": "compare",
            "Which alerts should we join before wildfire or hurricane season?": "alerts",
            "During an earthquake, should we run outside or stay where we are?": "earthquake",
        }
        for question, expected in cases.items():
            result = self.responder.answer(question, [])
            self.assertEqual(result["sources"][0]["id"], expected, question)
            self.assertEqual(result["mode"], "guidance")

    def test_wildfire_and_plan_and_overview(self):
        wildfire = self.responder.answer("When should we leave if there is a wildfire and smoke?", [])
        self.assertEqual(wildfire["sources"][0]["id"], "wildfire")
        self.assertIn("Ready, Set, Go", wildfire["reply"])

        plan = self.responder.answer("Where should we meet if we get separated?", [])
        self.assertEqual(plan["sources"][0]["id"], "plan")

        overview = self.responder.answer("How should we prepare across hazards?", [])
        self.assertEqual(overview["sources"][0]["id"], "overview")

    def test_unrelated_question_gets_the_fallback(self):
        result = self.responder.answer("What is the best pizza nearby?", [])
        self.assertEqual(result["sources"][0]["id"], "gap")
        self.assertIn("instead of guessing", result["reply"])

    @patch("chat.official.urllib.request.urlopen", side_effect=AssertionError("should not call the network"))
    def test_shelter_question_does_not_invent_a_location(self, _urlopen):
        result = self.responder.answer("Where is the nearest open shelter in Austin, Texas?", [])
        self.assertEqual(result["mode"], "unavailable")
        self.assertIn("will not invent", result["reply"])
        self.assertNotIn("Main Street", result["reply"])

    def test_live_alerts_require_a_place(self):
        result = self.responder.answer("Are there active weather alerts?", [])
        self.assertEqual(result["mode"], "unavailable")
        self.assertIn("city and state", result["reply"])

    @patch("chat.official.urllib.request.urlopen")
    def test_live_alerts_quote_the_weather_service(self, urlopen):
        urlopen.side_effect = [
            _FakeResponse(_CENSUS),
            _FakeResponse(_NWS),
        ]
        result = self.responder.answer("Are there active weather alerts in Austin, Texas?", [])
        self.assertEqual(result["mode"], "official")
        self.assertIn("Tornado Warning", result["reply"])
        self.assertIn("Take shelter now.", result["reply"])
        self.assertIn("does not list shelters", result["reply"])
        self.assertTrue(result["sources"][0]["retrieved_at"])
        self.assertEqual(result["sources"][0]["url"], "https://api.weather.gov/alerts/test")

    @patch("chat.responder.urllib.request.urlopen")
    def test_language_model_may_only_rephrase_the_source(self, urlopen):
        urlopen.return_value = _FakeResponse(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "Pack one gallon of water per person. This is general guidance, not a live report."
                            }
                        }
                    ]
                }
            )
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            os.environ.pop("RAG_BACKEND_URL", None)
            result = answer_question("What should go in a go-bag?", [])
        self.assertEqual(result["mode"], "llm")
        self.assertIn("one gallon", result["reply"])
        self.assertEqual(result["sources"][0]["id"], "gobag")
        sent = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertIn("never invent emergency information", sent["messages"][0]["content"])
        self.assertIn("one gallon", sent["messages"][1]["content"])


_CENSUS = json.dumps(
    {
        "results": [
            {
                "name": "Austin",
                "admin1": "Texas",
                "country_code": "US",
                "latitude": 30.2672,
                "longitude": -97.7431,
            }
        ]
    }
)

_NWS = json.dumps(
    {
        "features": [
            {
                "id": "https://api.weather.gov/alerts/test",
                "properties": {
                    "event": "Tornado Warning",
                    "headline": "Tornado Warning for Travis County",
                    "severity": "Extreme",
                    "senderName": "NWS Austin",
                    "sent": "2026-11-09T18:00:00-06:00",
                    "expires": "2026-11-09T18:45:00-06:00",
                    "instruction": "Take shelter now.",
                    "id": "https://api.weather.gov/alerts/test",
                },
            }
        ]
    }
)


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


if __name__ == "__main__":
    unittest.main()
