# Asset Extraction — Pure Python Function

Visual asset extraction as a plain Python function.  
No API. No server. Just call it, get the result.

---

## Project layout

```
asset-extraction-standalone/
├── agent.py       ← Public entry point  run_extraction() / run_extraction_sync()
├── tool.py        ← @tool extract_assets  (LangChain Deep Agent tool)
├── extraction.py  ← Bounding-box engine  (Gemini via OpenRouter)
├── config.py      ← Env-var settings
├── pyproject.toml
└── .env.example
```

---

## Quick start

```bash
cd asset-extraction-standalone
poetry install

cp .env.example .env
# → set OPENROUTER_API_KEY=sk-or-v1-...
```

---

## Usage

### Async (recommended)

```python
import asyncio
from agent import run_extraction

result = asyncio.run(run_extraction([
    {"path": "screenshot.png", "label": "company logo top-left"},
    {"url":  "https://…",      "label": "hero background photo"},
]))

for asset in result["assets"]:
    print(asset["label"], asset["status"], asset["box_2d"])
    # asset["data_url"] is a base64 PNG crop when status == "ok"
```

### Sync

```python
from agent import run_extraction_sync

result = run_extraction_sync([
    {"path": "shot.png", "label": "nav icon set"},
])
```

### From the command line

```bash
poetry run python agent.py screenshot.png "company logo"
poetry run python agent.py home.png "hero photo" icons.png "nav icon set"
```

Output is pretty-printed JSON.

---

## Input format

Each item in `images` is a plain `dict` with:

| Key | Type | Notes |
|-----|------|-------|
| `label` | `str` | ✅ required — what asset to find |
| `path` | `str` | local file path (absolute or relative) |
| `data_url` | `str` | `"data:image/png;base64,…"` |
| `url` | `str` | Public `http(s)://` URL — fetched automatically |

Provide exactly **one** of `path`, `data_url`, `url` per entry.

---

## Return value

```python
{
    "ok": True,
    "assets": [
        {
            "label":       "company logo top-left",
            "data_url":    "data:image/png;base64,…",  # PNG crop
            "status":      "ok",                        # "ok" | "missing" | "error"
            "box_2d":      [12.0, 18.0, 80.0, 210.0],  # ymin xmin ymax xmax (0-1000)
            "image_index": 1,                           # which input image (1-based)
        },
        {
            "label":       "missing widget",
            "data_url":    None,
            "status":      "missing",
            "box_2d":      None,
            "image_index": None,
        },
    ],
    # "error": "…"  ← only present when ok=False
}
```

---

## Embedding the tool in your own Deep Agent

`extract_assets` is a standard LangChain `@tool` — drop it into any agent:

```python
from deepagents import create_deep_agent
from tool import extract_assets

agent = create_deep_agent(
    model=your_chat_model,
    tools=[extract_assets],
    system_prompt="…",
)
```

---

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | ✅ | — | Your OpenRouter key |
| `ASSET_EXTRACTION_MODEL` | — | `google/gemini-3.6-flash` | Bounding-box detection model |
| `OPENROUTER_BASE_URL` | — | `https://openrouter.ai/api/v1` | Override for proxies |
| `OPENROUTER_APP_URL` | — | GitHub URL | Attribution in OR dashboard |
| `OPENROUTER_APP_TITLE` | — | `asset-extraction-standalone` | Attribution in OR dashboard |
