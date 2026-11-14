"""Harbor's knowledge base: the dataset the chatbot is allowed to answer from.

The dataset lives in a JSON file (``data/knowledge_base.json`` by default, or
the path in ``HARBOR_KB_PATH``) instead of in the code, so a developer can add,
edit, or remove documents without touching Python.

File format::

    {
      "version": 2,
      "documents": [
        {
          "id": "gobag",                      # unique, lowercase letters, digits, - or _
          "title": "One go-bag for several hazards",
          "body": "Text the chatbot may answer from...",
          "keywords": ["go bag", "emergency kit"],
          "source": "Harbor curated guidance", # optional, shown to the user
          "url": "https://www.ready.gov/kit",  # optional, shown to the user
          "updated": "2026-11-13",             # optional, YYYY-MM-DD
          "broad": false                       # optional, true = used only as a last resort
        }
      ]
    }

The app checks the file's modification time on each question and reloads it
when it changes. A file that fails validation is never loaded: Harbor keeps the
last good dataset and logs the problems, so a typo cannot empty the chatbot.

Check a file before deploying it::

    python -m chat.knowledge check
    python -m chat.knowledge check path/to/other.json

Other stories can build on this module: ``knowledge_base.reload()`` and
``knowledge_base.replace(documents)`` are the hooks for a clear/reload button
or an ingestion pipeline.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import sys
import threading
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_RESERVED_PREFIXES = ("gap", "nws")
_MAX_BODY = 20_000


class KnowledgeBaseError(Exception):
    """The dataset file is missing, unreadable, or invalid."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def kb_path() -> Path:
    raw = os.environ.get("HARBOR_KB_PATH", "").strip()
    return Path(raw) if raw else DEFAULT_PATH


def validate(data) -> tuple[list[dict], list[str]]:
    """Return (cleaned documents, problems). Documents are usable only if problems is empty."""
    problems: list[str] = []
    if not isinstance(data, dict):
        return [], ["The file must contain a JSON object with a \"documents\" list."]
    raw_docs = data.get("documents")
    if not isinstance(raw_docs, list):
        return [], ["\"documents\" must be a list."]
    if not raw_docs:
        problems.append("\"documents\" is empty. Harbor needs at least one document.")

    cleaned: list[dict] = []
    seen: set[str] = set()
    broad_count = 0
    for index, item in enumerate(raw_docs):
        where = f"document #{index + 1}"
        if not isinstance(item, dict):
            problems.append(f"{where}: must be an object.")
            continue
        doc_id = item.get("id")
        if isinstance(doc_id, str):
            where = f"document \"{doc_id}\""
        if not isinstance(doc_id, str) or not _ID_PATTERN.match(doc_id):
            problems.append(f"{where}: \"id\" must be lowercase letters, digits, - or _ (max 64).")
            continue
        if doc_id.startswith(_RESERVED_PREFIXES):
            problems.append(f"{where}: ids starting with gap or nws are reserved.")
        if doc_id in seen:
            problems.append(f"{where}: duplicate id.")
        seen.add(doc_id)

        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            problems.append(f"{where}: \"title\" is required.")
            title = ""
        body = item.get("body")
        if not isinstance(body, str) or not body.strip():
            problems.append(f"{where}: \"body\" is required.")
            body = ""
        elif len(body) > _MAX_BODY:
            problems.append(f"{where}: \"body\" is longer than {_MAX_BODY} characters. Split it up.")

        keywords = item.get("keywords", [])
        if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
            problems.append(f"{where}: \"keywords\" must be a list of strings.")
            keywords = []
        keywords = [_normalize_keyword(k) for k in keywords if k.strip()]
        if not keywords:
            problems.append(f"{where}: add at least one keyword so questions can find it.")

        doc = {
            "id": doc_id,
            "title": title.strip(),
            "body": body.strip(),
            "keywords": keywords,
            "source": "Harbor curated guidance",
            "broad": False,
        }
        source = item.get("source")
        if source is not None:
            if isinstance(source, str) and source.strip():
                doc["source"] = source.strip()
            else:
                problems.append(f"{where}: \"source\" must be text.")
        url = item.get("url")
        if url is not None:
            if isinstance(url, str) and url.startswith(("https://", "http://")):
                doc["url"] = url
            else:
                problems.append(f"{where}: \"url\" must start with https:// or http://.")
        updated = item.get("updated")
        if updated is not None:
            try:
                date.fromisoformat(str(updated))
                doc["updated"] = str(updated)
            except ValueError:
                problems.append(f"{where}: \"updated\" must be a date like 2026-11-13.")
        broad = item.get("broad", False)
        if not isinstance(broad, bool):
            problems.append(f"{where}: \"broad\" must be true or false.")
        elif broad:
            doc["broad"] = True
            broad_count += 1
        unknown = set(item) - {"id", "title", "body", "keywords", "source", "url", "updated", "broad"}
        if unknown:
            problems.append(f"{where}: unknown field(s) {', '.join(sorted(unknown))}.")
        cleaned.append(doc)

    if broad_count > 1:
        problems.append("Only one document may set \"broad\": true.")
    return cleaned, problems


def load_file(path: Path) -> list[dict]:
    """Read and validate a dataset file. Raises KnowledgeBaseError on any problem."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise KnowledgeBaseError([f"Could not read {path}: {exc.strerror or exc}"]) from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseError([f"{path} is not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}"]) from exc
    documents, problems = validate(data)
    if problems:
        raise KnowledgeBaseError(problems)
    return documents


class KnowledgeBase:
    """Thread-safe holder for the current dataset, reloaded when the file changes."""

    def __init__(self, path: Path | None = None):
        self._explicit_path = Path(path) if path else None
        self._lock = threading.Lock()
        self._documents: list[dict] = []
        self._loaded_path: Path | None = None
        self._mtime: float | None = None
        self._last_error: list[str] = []

    @property
    def path(self) -> Path:
        return self._explicit_path or kb_path()

    def documents(self) -> list[dict]:
        """Current documents (copies). Reloads first if the file changed on disk."""
        self._reload_if_changed()
        with self._lock:
            return copy.deepcopy(self._documents)

    def reload(self) -> dict:
        """Force a reload from disk. On failure the previous dataset stays in place."""
        path = self.path
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = None
        try:
            documents = load_file(path)
        except KnowledgeBaseError as exc:
            with self._lock:
                self._last_error = exc.problems
                self._mtime = mtime
                self._loaded_path = path
            logger.error("Knowledge base not loaded from %s: %s", path, exc)
            return self.status()
        with self._lock:
            self._documents = documents
            self._mtime = mtime
            self._loaded_path = path
            self._last_error = []
        logger.info("Knowledge base loaded: %d documents from %s", len(documents), path)
        return self.status()

    def replace(self, documents: list[dict]) -> dict:
        """Swap in documents from another source (e.g. an ingestion step) after validation."""
        cleaned, problems = validate({"documents": documents})
        if problems:
            raise KnowledgeBaseError(problems)
        with self._lock:
            self._documents = cleaned
            self._last_error = []
        return self.status()

    def status(self) -> dict:
        with self._lock:
            return {
                "path": str(self._loaded_path or self.path),
                "documents": len(self._documents),
                "ids": [doc["id"] for doc in self._documents],
                "errors": list(self._last_error),
            }

    def _reload_if_changed(self) -> None:
        path = self.path
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = None
        with self._lock:
            unchanged = path == self._loaded_path and mtime == self._mtime
        if not unchanged:
            self.reload()


def _normalize_keyword(text: str) -> str:
    # Must match how questions are normalized in chat.responder, or keywords will never hit.
    text = text.lower().replace("’", "'").replace("'", "")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


knowledge_base = KnowledgeBase()


def _main(argv: list[str]) -> int:
    if len(argv) < 1 or argv[0] != "check":
        print("Usage: python -m chat.knowledge check [path]")
        return 2
    path = Path(argv[1]) if len(argv) > 1 else kb_path()
    try:
        documents = load_file(path)
    except KnowledgeBaseError as exc:
        print(f"FAILED: {path}")
        for problem in exc.problems:
            print(f"  - {problem}")
        return 1
    print(f"OK: {path} has {len(documents)} documents")
    for doc in documents:
        extra = " (broad)" if doc.get("broad") else ""
        print(f"  - {doc['id']}: {doc['title']}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
