"""Tests for story 7 (clean and preprocess data before ingestion)."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from chat import knowledge
from chat.knowledge import KnowledgeBase, KnowledgeBaseError
from chat.preprocess import clean_text, preprocess


def _doc(**overrides):
    doc = {
        "id": "test-doc",
        "title": "Test document",
        "body": "Store a spare key with a trusted neighbor before the storm season.",
        "keywords": ["spare key"],
    }
    doc.update(overrides)
    return doc


class CleanTextTests(unittest.TestCase):
    def test_html_is_removed_and_entities_decoded(self):
        raw = "<h2>Flood&nbsp;safety</h2><p>Turn around, don&#39;t drown.</p><script>track()</script>"
        self.assertEqual(clean_text(raw, keep_paragraphs=True), "Flood safety\n\nTurn around, don't drown.")

    def test_comparison_signs_are_not_mistaken_for_tags(self):
        self.assertEqual(clean_text("winds < 40 mph and > 20 mph"), "winds < 40 mph and > 20 mph")

    def test_quotes_control_characters_and_whitespace(self):
        raw = "“Go”   bag​\x00 ’s Ａ"
        self.assertEqual(clean_text(raw), "\"Go\" bag 's A")

    def test_paragraphs_are_kept_only_for_body_text(self):
        raw = "First   line\nstill first.\n\n\n\nSecond."
        self.assertEqual(clean_text(raw, keep_paragraphs=True), "First line still first.\n\nSecond.")
        self.assertEqual(clean_text(raw), "First line still first. Second.")


class PreprocessTests(unittest.TestCase):
    def test_documents_are_cleaned_and_changes_reported(self):
        data, changes = preprocess(
            {"documents": [_doc(title="  <b>Spare</b> key ", keywords=["Spare Key", "spare key", " "])]}
        )
        doc = data["documents"][0]
        self.assertEqual(doc["title"], "Spare key")
        self.assertEqual(doc["keywords"], ["spare key"])
        self.assertEqual(changes, ['document "test-doc": cleaned title, keywords'])

    def test_clean_document_reports_nothing(self):
        data, changes = preprocess({"documents": [_doc()]})
        self.assertEqual(data["documents"], [_doc()])
        self.assertEqual(changes, [])

    def test_duplicate_body_is_dropped(self):
        copy = _doc(id="copy", body="  STORE a spare key with a trusted neighbor before the storm season. ")
        data, changes = preprocess({"documents": [_doc(), copy]})
        self.assertEqual([d["id"] for d in data["documents"]], ["test-doc"])
        self.assertIn('document "copy": removed, same body as "test-doc"', changes)

    def test_wrong_shapes_pass_through_for_validation(self):
        self.assertEqual(preprocess([]), ([], []))
        data, _ = preprocess({"documents": ["not a dict", _doc(body=5)]})
        self.assertEqual(data["documents"][0], "not a dict")
        self.assertEqual(data["documents"][1]["body"], 5)

    def test_markup_only_body_fails_validation(self):
        with self.assertRaises(KnowledgeBaseError) as caught:
            KnowledgeBase().replace([_doc(body="<div><br></div>")])
        self.assertTrue(any("body" in problem for problem in caught.exception.problems))


class IngestionUsesPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "kb.json"
        self.path.write_text(
            json.dumps({"documents": [_doc(body="<p>Store a spare key&nbsp;with a neighbor.</p>")]}),
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_file_load_is_cleaned(self):
        kb = KnowledgeBase(self.path)
        self.assertEqual(kb.documents()[0]["body"], "Store a spare key with a neighbor.")
        self.assertEqual(kb.status()["cleaned"], ['document "test-doc": cleaned body'])

    def test_replace_is_cleaned(self):
        status = KnowledgeBase(self.path).replace([_doc(title="<i>Key</i>")])
        self.assertEqual(status["cleaned"], ['document "test-doc": cleaned title'])

    def test_check_command_lists_changes(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = knowledge._main(["check", str(self.path)])
        self.assertEqual(code, 0)
        self.assertIn("Preprocessing changed:", out.getvalue())
        self.assertIn('document "test-doc": cleaned body', out.getvalue())


if __name__ == "__main__":
    unittest.main()
