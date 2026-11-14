"""Tests for story 3 (answers grounded in the dataset) and story 5 (editable dataset)."""

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from chat import knowledge
from chat.grounding import is_supported, unsupported_sentences
from chat.knowledge import KnowledgeBase, KnowledgeBaseError, load_file, validate
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


def _doc(**overrides):
    doc = {
        "id": "test-doc",
        "title": "Test document",
        "body": "Store a spare key with a trusted neighbor before the storm season.",
        "keywords": ["spare key"],
    }
    doc.update(overrides)
    return doc


def _model_reply(content) -> _FakeResponse:
    if not isinstance(content, str):
        content = json.dumps(content)
    return _FakeResponse(json.dumps({"choices": [{"message": {"content": content}}]}))


class _TempDataset(unittest.TestCase):
    """Points Harbor at a temporary copy of the dataset for each test."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "kb.json"
        shutil.copy(knowledge.DEFAULT_PATH, self.path)
        self.env = patch.dict(os.environ, {"HARBOR_KB_PATH": str(self.path)})
        self.env.start()
        os.environ.pop("RAG_BACKEND_URL", None)
        os.environ.pop("OPENAI_API_KEY", None)

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")
        # Make sure the modification time moves even on coarse filesystems.
        stamp = time.time() + 5
        os.utime(self.path, (stamp, stamp))

    def read(self):
        return json.loads(self.path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- story 5


class DatasetFileTests(unittest.TestCase):
    def test_shipped_dataset_is_valid(self):
        documents = load_file(knowledge.DEFAULT_PATH)
        ids = [d["id"] for d in documents]
        self.assertEqual(len(ids), len(set(ids)))
        for expected in ("gobag", "compare", "alerts", "earthquake", "wildfire", "plan", "overview"):
            self.assertIn(expected, ids)
        self.assertEqual([d["id"] for d in documents if d["broad"]], ["overview"])

    def test_validation_reports_every_problem(self):
        _docs, problems = validate(
            {
                "documents": [
                    _doc(id="Bad ID"),
                    _doc(id="dup"),
                    _doc(id="dup", body=""),
                    _doc(id="nokeys", keywords=[]),
                    _doc(id="badurl", url="ftp://x"),
                    _doc(id="baddate", updated="13/11/2026"),
                    _doc(id="gap-shelter"),
                    _doc(id="typo", bodyy="x"),
                ]
            }
        )
        text = " | ".join(problems)
        for fragment in ("lowercase", "duplicate", "\"body\" is required", "keyword",
                         "https://", "date like", "reserved", "unknown field"):
            self.assertIn(fragment, text)

    def test_empty_or_wrong_shape_is_rejected(self):
        self.assertTrue(validate({"documents": []})[1])
        self.assertTrue(validate([])[1])
        self.assertTrue(validate({"docs": []})[1])

    def test_keywords_are_normalized_like_questions(self):
        documents, problems = validate({"documents": [_doc(keywords=["Go-Bag", "  What’s  NEW? "])]})
        self.assertEqual(problems, [])
        self.assertEqual(documents[0]["keywords"], ["go bag", "whats new"])

    def test_check_command(self):
        self.assertEqual(knowledge._main(["check", str(knowledge.DEFAULT_PATH)]), 0)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write("{not json")
        try:
            self.assertEqual(knowledge._main(["check", handle.name]), 1)
        finally:
            os.unlink(handle.name)


class DatasetUpdateTests(_TempDataset):
    def test_new_document_is_answered_without_restarting(self):
        before = answer_question("Where should we keep a spare key?", [])
        self.assertNotEqual(before["sources"][0].get("id"), "test-doc")

        data = self.read()
        data["documents"].append(
            _doc(source="County Emergency Office", url="https://example.gov/keys", updated="2026-11-13")
        )
        self.write(data)

        after = answer_question("Where should we keep a spare key?", [])
        self.assertEqual(after["mode"], "guidance")
        self.assertEqual(after["sources"][0]["id"], "test-doc")
        self.assertIn("trusted neighbor", after["reply"])
        self.assertIn("County Emergency Office", after["sources"][0]["detail"])
        self.assertIn("updated 2026-11-13", after["sources"][0]["detail"])
        self.assertEqual(after["sources"][0]["url"], "https://example.gov/keys")

    def test_edited_document_changes_the_answer(self):
        data = self.read()
        for doc in data["documents"]:
            if doc["id"] == "gobag":
                doc["body"] = "Revised go-bag guidance: pack two gallons of water per person per day."
        self.write(data)
        result = answer_question("What should go in a go-bag?", [])
        self.assertIn("two gallons", result["reply"])
        self.assertNotIn("one gallon", result["reply"])

    def test_removed_document_is_no_longer_used(self):
        data = self.read()
        data["documents"] = [d for d in data["documents"] if d["id"] != "wildfire"]
        self.write(data)
        result = answer_question("What does ready set go mean for wildfire?", [])
        for source in result["sources"]:
            self.assertNotEqual(source.get("id"), "wildfire")

    def test_broken_file_keeps_last_good_dataset(self):
        self.assertEqual(answer_question("What should go in a go-bag?", [])["sources"][0]["id"], "gobag")
        self.path.write_text("{ this is broken", encoding="utf-8")
        stamp = time.time() + 10
        os.utime(self.path, (stamp, stamp))
        result = answer_question("What should go in a go-bag?", [])
        self.assertEqual(result["sources"][0]["id"], "gobag")
        status = knowledge.knowledge_base.status()
        self.assertTrue(status["errors"])
        self.assertEqual(status["documents"], 7)

    def test_reload_and_replace_hooks(self):
        kb = KnowledgeBase(self.path)
        self.assertEqual(kb.reload()["documents"], 7)
        self.assertEqual(kb.replace([_doc()])["ids"], ["test-doc"])
        with self.assertRaises(KnowledgeBaseError):
            kb.replace([_doc(id="")])
        self.assertEqual(kb.status()["ids"], ["test-doc"])


# ---------------------------------------------------------------- story 3


class GroundingCheckTests(unittest.TestCase):
    SOURCE = (
        "For each person, a widely used planning baseline is one gallon of water per day "
        "for at least three days, plus food that does not need a stove."
    )

    def test_faithful_rewording_passes(self):
        self.assertTrue(is_supported("Plan on one gallon of water per person each day, for three days.", [self.SOURCE]))

    def test_invented_number_fails(self):
        self.assertFalse(is_supported("Store one gallon of water per person for 14 days.", [self.SOURCE]))

    def test_invented_fact_fails(self):
        bad = unsupported_sentences(
            "Plan on one gallon of water per day. The Red Cross shelter at Lincoln High School is open tonight.",
            [self.SOURCE],
        )
        self.assertEqual(len(bad), 1)
        self.assertIn("Lincoln", bad[0])

    def test_empty_answer_or_no_sources_fails(self):
        self.assertFalse(is_supported("", [self.SOURCE]))
        self.assertFalse(is_supported("One gallon of water.", []))


class GroundedModelAnswerTests(_TempDataset):
    @patch("chat.responder.urllib.request.urlopen")
    def test_model_answer_with_invented_detail_is_replaced_by_source(self, urlopen):
        urlopen.return_value = _model_reply(
            {
                "passage_ids": ["gobag"],
                "lookup_alerts": False,
                "answer": "Pack 10 gallons of water and go to the shelter on 5th Avenue, which FEMA has opened.",
            }
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = answer_question("What should go in a go-bag?", [])
        self.assertNotIn("5th Avenue", result["reply"])
        self.assertNotIn("10 gallons", result["reply"])
        self.assertIn("one gallon of water per day", result["reply"])
        self.assertIn("source text directly", result["notice"])
        self.assertEqual(result["sources"][0]["id"], "gobag")

    @patch("chat.responder.urllib.request.urlopen")
    def test_model_answer_that_stays_in_the_source_is_kept(self, urlopen):
        answer = "Drop, Cover, and Hold On until the shaking stops. Do not run outside during the shaking."
        urlopen.return_value = _model_reply({"passage_ids": ["earthquake"], "lookup_alerts": False, "answer": answer})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = answer_question("The room is shaking, what now?", [])
        self.assertEqual(result["mode"], "llm")
        self.assertTrue(result["reply"].startswith(answer))
        self.assertIn("checked against", result["notice"])

    @patch("chat.responder.urllib.request.urlopen")
    def test_free_text_model_reply_is_checked_too(self, urlopen):
        urlopen.return_value = _model_reply("Buy a generator from Home Depot and stock 30 days of canned soup.")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = answer_question("What should go in a go-bag?", [])
        self.assertNotIn("Home Depot", result["reply"])
        self.assertEqual(result["mode"], "guidance")
        self.assertIn("one gallon", result["reply"])

    @patch("chat.responder.urllib.request.urlopen")
    def test_model_cannot_cite_a_document_that_is_not_in_the_dataset(self, urlopen):
        urlopen.return_value = _model_reply(
            {"passage_ids": ["made-up-doc"], "lookup_alerts": False, "answer": "Invented guidance."}
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = answer_question("What should go in a go-bag?", [])
        self.assertNotIn("Invented guidance", result["reply"])
        self.assertNotIn("made-up-doc", [s.get("id") for s in result["sources"]])

    def test_every_answer_has_a_source(self):
        questions = [
            "What should go in a go-bag?",
            "How is leaving for a flood different from leaving for a hurricane?",
            "Where is the nearest shelter?",
            "Tell me a joke about cats",
            "How should I prepare?",
        ]
        for question in questions:
            with self.subTest(question=question):
                result = answer_question(question, [])
                self.assertTrue(result["sources"], question)


class GroundedRetrievalServiceTests(_TempDataset):
    def _ask(self, urlopen, payload):
        urlopen.return_value = _FakeResponse(json.dumps(payload))
        with patch.dict(os.environ, {"RAG_BACKEND_URL": "http://rag.internal/answer"}):
            return answer_question("What should we pack?", [])

    @patch("chat.responder.urllib.request.urlopen")
    def test_answer_without_sources_is_withheld(self, urlopen):
        result = self._ask(urlopen, {"reply": "Pack 40 gallons of water.", "sources": []})
        self.assertEqual(result["mode"], "unavailable")
        self.assertNotIn("40 gallons", result["reply"])
        self.assertIn("withheld", result["notice"])

    @patch("chat.responder.urllib.request.urlopen")
    def test_supported_answer_is_shown(self, urlopen):
        result = self._ask(
            urlopen,
            {
                "reply": "Pack one gallon of water per person per day for three days.",
                "sources": [{"title": "Ready.gov kit", "text": GroundingCheckTests.SOURCE}],
            },
        )
        self.assertEqual(result["mode"], "rag")
        self.assertIn("one gallon", result["reply"])
        self.assertIsNone(result["notice"])

    @patch("chat.responder.urllib.request.urlopen")
    def test_unsupported_answer_is_replaced_by_source_text(self, urlopen):
        result = self._ask(
            urlopen,
            {
                "reply": "Pack 40 gallons of water. Shelters are open at City Hall.",
                "sources": [{"title": "Ready.gov kit", "text": GroundingCheckTests.SOURCE}],
            },
        )
        self.assertNotIn("City Hall", result["reply"])
        self.assertIn("Ready.gov kit", result["reply"])
        self.assertIn("source text directly", result["notice"])

    @patch("chat.responder.urllib.request.urlopen")
    def test_titles_only_answer_is_flagged_as_unchecked(self, urlopen):
        result = self._ask(urlopen, {"reply": "Some answer.", "sources": ["Article A"]})
        self.assertEqual(result["mode"], "rag")
        self.assertIn("could not check", result["notice"])


if __name__ == "__main__":
    unittest.main()
