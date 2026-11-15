"""Tests for story 8 (prompts chosen at runtime from the type of question)."""

import json
import os
import unittest
from unittest.mock import patch

from chat.prompts import INTENTS, URGENT_LINE, build_system_prompt, classify_intent, shape_passage
from chat.responder import answer_question


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _model_reply(content: str) -> _FakeResponse:
    return _FakeResponse(json.dumps({"choices": [{"message": {"content": content}}]}))


class ClassifyIntentTests(unittest.TestCase):
    def test_known_questions(self):
        cases = {
            "How much water per person should we store?": "factual",
            "What is a flash flood?": "factual",
            "What should go in one go-bag for floods and earthquakes?": "steps",
            "How do I get ready for wildfire season?": "steps",
            "Why should we not run outside during an earthquake?": "explanation",
            "What's the difference between a flood watch and a warning?": "explanation",
            "Write a text to my family about where we meet": "creative",
            "Help me plan for my grandmother and our dog": "creative",
            "We are trapped and the water is rising": "urgent",
            "Help, my neighbor is injured, what should I do?": "urgent",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(classify_intent(question), expected)

    def test_phrases_match_whole_words_only(self):
        # "why" inside "anywhere" and "draft" inside "drafty" must not count.
        self.assertEqual(classify_intent("Is anywhere drafty safe?"), "factual")


class BuildPromptTests(unittest.TestCase):
    def test_every_intent_keeps_safety_rules_and_adds_its_own_style(self):
        prompts = {intent: build_system_prompt(intent) for intent in INTENTS}
        self.assertEqual(len(set(prompts.values())), len(INTENTS))
        for intent, prompt in prompts.items():
            self.assertIn("never invent emergency information", prompt)
            self.assertIn(INTENTS[intent]["instructions"], prompt)

    def test_passage_is_shaped_without_new_words(self):
        body = "Water is one gallon a day.\n\nPack a flashlight. Add batteries."
        self.assertEqual(shape_passage(body, "steps", "what to pack"), "1. Water is one gallon a day.\n2. Pack a flashlight.\n3. Add batteries.")
        self.assertEqual(shape_passage(body, "factual", "how much water"), "Water is one gallon a day.")
        self.assertEqual(shape_passage(body, "explanation", "why"), body)


class RuntimeSelectionTests(unittest.TestCase):
    def setUp(self):
        self._env = patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("RAG_BACKEND_URL", None)
        os.environ.pop("OPENAI_API_KEY", None)

    def tearDown(self):
        self._env.stop()

    @patch("chat.responder.urllib.request.urlopen")
    def test_model_receives_the_prompt_for_the_detected_intent(self, urlopen):
        os.environ["OPENAI_API_KEY"] = "test-key"
        for question, intent in (
            ("What should go in a go-bag?", "steps"),
            ("Why should we not run outside during an earthquake?", "explanation"),
        ):
            with self.subTest(intent=intent):
                urlopen.return_value = _model_reply('{"passage_ids": [], "answer": ""}')
                result = answer_question(question, [])
                sent = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
                self.assertEqual(sent["messages"][0]["content"], build_system_prompt(intent))
                self.assertIn(INTENTS[intent]["label"], sent["messages"][1]["content"])
                self.assertEqual(result["intent"], intent)

    def test_keyword_path_shapes_the_answer_by_intent(self):
        steps = answer_question("What should go in one go-bag for floods, hurricanes, wildfires, and earthquakes?")
        self.assertEqual(steps["intent_label"], "Step-by-step")
        self.assertTrue(steps["reply"].startswith("1. "))
        # Shaping changes the reply only; the full passage is still the cited source.
        self.assertIn("\n\n", steps["sources"][0]["text"])

    def test_urgent_questions_lead_with_emergency_services(self):
        result = answer_question("We are trapped and the water is rising")
        self.assertEqual(result["intent"], "urgent")
        self.assertTrue(result["reply"].startswith(URGENT_LINE))

    @patch("chat.responder.urllib.request.urlopen")
    def test_retrieval_service_is_told_the_intent(self, urlopen):
        os.environ["RAG_BACKEND_URL"] = "http://rag.test/api/answer"
        urlopen.return_value = _FakeResponse(json.dumps({"reply": "Pack water.", "sources": [{"title": "Kit", "text": "Pack water."}]}))
        answer_question("What should go in a go-bag?")
        sent = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(sent["intent"], "steps")


if __name__ == "__main__":
    unittest.main()
