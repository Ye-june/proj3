# Harbor

Harbor is a browser chat for a problem that shows up across disasters, not inside just one of them. Households facing floods, hurricanes, wildfires, and earthquakes are handed separate advice for each hazard. A lot of the useful preparation is shared: one go-bag, one alert signup, one way to find each other. A few decisions are not shared, and mixing them up is dangerous. Running outside during earthquake shaking, driving into floodwater, waiting out a hurricane evacuation order, and leaving a wildfire too late are different situations.

Harbor answers only from a retrieved source, and it says when that source is missing. Active weather alerts come from the National Weather Service for a U.S. city and state. General safety answers come from a curated guidance set and are labeled as guidance, not as a live incident report. Shelter locations, road closures, power outages, and evacuation orders are not invented.

An optional OpenAI-compatible Chat Completions API can rephrase that retrieved text. Without `OPENAI_API_KEY`, the source wording is shown directly. `RAG_BACKEND_URL` still forwards questions to a separate retrieval service when the team connects one.

## What you can do

- Open the app in a desktop or mobile browser. Nothing to install.
- Ask a question, or start from one of the examples.
- Read your messages and Harbor's replies in one scrolling conversation.
- See a loading state while an answer is on the way, and a readable error if the request fails.

## Project layout

| Path | Role |
| --- | --- |
| `app.py` | Flask routes: the page, `/health`, and `POST /api/chat` |
| `wsgi.py` | Production entry point for Gunicorn |
| `chat/responder.py` | Grounded answers, the optional language-model rephrase, and the RAG hook |
| `chat/official.py` | City lookup and National Weather Service alerts |
| `chat/policy.py` | The rule against inventing emergency facts |
| `templates/index.html` | Chat page |
| `static/css/style.css` | Layout and visual design |
| `static/js/chat.js` | Sending messages and rendering the thread |
| `render.yaml`, `Procfile`, `runtime.txt` | Hosting configuration |
| `tests/test_app.py` | Route and preview-routing tests |

## Run locally

Python 3.11 or newer is required. On Windows, `py -3.11` is a reliable way to pick 3.11 if several versions are installed.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
python app.py
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

`FLASK_DEBUG=1` in `.env` turns on the local debugger. Leave that off on any shared or public machine. Gunicorn is the production server. It does not run on Windows, which is why local development uses `python app.py`.

### Tests

From the project root, with the virtual environment active:

```powershell
python -m unittest discover -s tests -v
```

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `SECRET_KEY` | Yes, in production | Flask secret. Use a long random string. `render.yaml` can generate one. |
| `PORT` | Set by the host | Port to bind. Defaults to `5000` when you run `python app.py`. |
| `FLASK_DEBUG` | No | Set to `1` only for local debugging. |
| `RAG_BACKEND_URL` | No | When set, chat questions are POSTed here instead of using the preview set. |
| `RAG_TIMEOUT` | No | Seconds to wait for the RAG service, alert feeds, or language model. Defaults to 30. |
| `OPENAI_API_KEY` | No | When set, retrieved text is rephrased through Chat Completions. |
| `OPENAI_BASE_URL` | No | OpenAI-compatible base URL. Defaults to `https://api.openai.com/v1`. |
| `OPENAI_MODEL` | No | Model name. Defaults to `gpt-4o-mini`. |

The app reads a local `.env` file if one exists, and does not override variables the host has already set. Do not commit `.env`.

## Connect the RAG backend later

Leave the web app as it is. When retrieval exists, either:

1. Deploy the RAG service and set `RAG_BACKEND_URL` to its answer endpoint, or
2. Replace the body of `answer_question()` in `chat/responder.py` with a direct call to that module.

Expected HTTP contract:

```http
POST /api/answer
Content-Type: application/json

{"message": "What belongs in a go-bag?", "history": [{"role": "user", "content": "Earlier turn"}]}
```

```json
{"reply": "Grounded answer text.", "sources": ["Document title", {"title": "Another source"}]}
```

`reply` may also be named `answer` or `response`. The page shows source titles and the time they were checked under the assistant message. While `RAG_BACKEND_URL` is empty, Harbor uses its own retrieval. `mode` is `official` for a weather-alert lookup, `guidance` for curated safety text, `unavailable` when a live source is missing, or `llm` when a model rephrased the retrieved text. The header reads **Sources required** unless a key or retrieval URL is configured.

Conversation history stays in the browser tab. Each request sends the recent turns so a future model can use them. Harbor does not store questions on the server.

## Deploy on Render

The app is prepared for [Render](https://render.com), which can run this Python process on a public URL. A Render account and a hosted Git remote are required. Those credentials were not available in this workspace, so the public URL still has to be created from the steps below.

1. Push this project to GitHub or another Git host Render can read.
2. In the Render dashboard, choose **New +** → **Blueprint**, and point it at the repository. Render will read `render.yaml`.
3. Confirm the service name, the free plan (change it if your account has no free web services), and the generated `SECRET_KEY`.
4. Leave `RAG_BACKEND_URL` empty until retrieval is deployed. Set it later under **Environment** without changing the web code.
5. Deploy. The start command is `gunicorn wsgi:app --bind 0.0.0.0:$PORT --timeout 90`.
6. Open the service URL Render assigns, such as `https://harbor-xxxx.onrender.com`. A sleeping free instance can take a minute to wake up. `/health` should return `{"status": "ok"}`.

To create the service without the blueprint, use **New +** → **Web Service**, set the runtime to Python, the build command to `pip install -r requirements.txt`, and the start command above. Add `SECRET_KEY` yourself. `runtime.txt` and the `Procfile` are there for hosts that read those files instead of `render.yaml`.

Other Python hosts work the same way: install `requirements.txt`, set `PORT` and `SECRET_KEY`, and start Gunicorn with `wsgi:app`. Bind to `0.0.0.0`, not `127.0.0.1`.

## Still to connect

- The approved article dataset and the retrieval pipeline. The preview passages in `chat/responder.py` are labeled placeholder content.
- A public host account, if the URL has not been created yet.
- Administrative pages for ingestion. They are outside this user story and are not in this app.
