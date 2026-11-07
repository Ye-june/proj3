"""Tests for the Harbor web app and the preview responder."""

import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from app import create_app
from chat.responder import PlaceholderResponder


class HarborAppTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("RAG_BACKEND_URL", None)
        self.app = create_app()
        self.client = self.app.test_client()

    def test_homepage_renders_chat_shell(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("Harbor", page)
        self.assertIn('id="composer"', page)
        self.assertIn("Preview knowledge", page)
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
        self.assertEqual(body["mode"], "placeholder")
        self.assertIn("one gallon", body["reply"])
        self.assertEqual(body["sources"][0]["id"], "gobag")
        self.assertIn("Preview", body["notice"])

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
            self.assertEqual(result["mode"], "placeholder")

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
        self.assertEqual(result["sources"][0]["id"], "preview")
        self.assertIn("plan ahead", result["reply"])


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
