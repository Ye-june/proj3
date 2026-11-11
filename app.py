"""Harbor web application.

Serves the chat interface and a small JSON API. Retrieval lives behind
``chat.responder.answer_question`` so this file stays the web layer only.
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from chat.responder import ResponderError, answer_question


def load_local_env(path: str | None = None) -> None:
    """Load a local ``.env`` file without overriding variables already set.

    This stays dependency-free so a host that injects real environment
    variables does not need python-dotenv installed.
    """
    env_path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


def create_app() -> Flask:
    load_local_env()
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-only-not-for-production")
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.json.ensure_ascii = False

    @app.get("/")
    def index():
        mode = _page_mode()
        return render_template("index.html", mode=mode)

    @app.get("/admin/knowledge-base")
    def knowledge_base():
        return render_template("knowledge_base.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Enter a question to continue."}), 400

        message = payload.get("message", "")
        if not isinstance(message, str) or not message.strip():
            return jsonify({"error": "Enter a question to continue."}), 400
        message = message.strip()
        if len(message) > 2000:
            return jsonify({"error": "Please keep questions under 2000 characters."}), 400

        history = _clean_history(payload.get("history"))
        try:
            result = answer_question(message, history)
        except ResponderError as exc:
            return jsonify({"error": str(exc)}), 502
        except Exception:
            app.logger.exception("chat request failed")
            return jsonify({"error": "Harbor could not answer just now. Please try again."}), 500
        return jsonify(result)

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "That message is too large."}), 413

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found."}), 404
        return "That page is not part of Harbor.", 404

    return app


def _page_mode() -> str:
    if os.environ.get("RAG_BACKEND_URL", "").strip():
        return "rag"
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "llm"
    return "grounded"


def _clean_history(history) -> list[dict]:
    if not isinstance(history, list):
        return []
    cleaned = []
    for item in history[-8:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content[:4000]})
    return cleaned


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
