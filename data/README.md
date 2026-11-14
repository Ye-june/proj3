# Harbor dataset

`knowledge_base.json` is everything Harbor's chatbot is allowed to answer from
(apart from live National Weather Service alerts). If a fact is not in this
file, Harbor will not state it.

## Add or edit a document

1. Open `knowledge_base.json` and add an entry to `"documents"` (or edit one):

   ```json
   {
     "id": "power-safety",
     "title": "Generator safety during an outage",
     "body": "Run generators outdoors, at least 20 feet from windows and doors...",
     "keywords": ["generator", "carbon monoxide", "power outage safety"],
     "source": "Ready.gov",
     "url": "https://www.ready.gov/power-outages",
     "updated": "2026-11-13"
   }
   ```

   | Field | Required | Notes |
   |---|---|---|
   | `id` | yes | Unique. Lowercase letters, digits, `-` or `_`. Cannot start with `gap` or `nws`. |
   | `title` | yes | Shown to the user as the source title. |
   | `body` | yes | The text Harbor answers from. Up to 20,000 characters; split longer material. |
   | `keywords` | yes | Phrases a question might contain. Longer phrases score higher. |
   | `source` | no | Who the information comes from. Defaults to "Harbor curated guidance". |
   | `url` | no | Link shown with the source. Must start with `https://` or `http://`. |
   | `updated` | no | `YYYY-MM-DD`. Shown with the source so users can see how current it is. |
   | `broad` | no | `true` for the single catch-all overview used only when nothing else matches. |

2. Check the file:

   ```
   python -m chat.knowledge check
   ```

   It lists every problem at once (duplicate ids, missing fields, bad dates, typos in field names).

3. Run the tests: `python -m pytest`

4. Commit and push. Render redeploys with the new dataset.

## How updates take effect

- **Locally:** Harbor notices the file changed and reloads it on the next question. No restart needed.
- **On Render:** the file ships with the code, so changes go live on the next deploy.
- **Bad file:** if the file fails validation, Harbor keeps the last good dataset and logs the problems. A typo cannot empty the chatbot.

To use a different file, set `HARBOR_KB_PATH`.

## For other stories

`chat/knowledge.py` exposes `knowledge_base.reload()`, `knowledge_base.replace(documents)` and
`knowledge_base.status()`. The admin clear/reload buttons (story 11) and ingestion pipelines
(stories 6 and 7) can call these instead of writing their own loader. `replace()` runs the same
validation as the file.
