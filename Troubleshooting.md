# Troubleshooting

This fork uses a single **OpenRouter** API key for every model. Create a key at https://openrouter.ai/keys, add credits, and put it in `backend/.env` as `OPENROUTER_API_KEY` (or paste it in the Settings dialog).

Common errors:

- **"No OpenRouter API key found"** — the key is missing from both `.env` and the settings dialog. Restart the backend after editing `.env`.
- **"Incorrect OpenRouter key"** — the key was rejected (HTTP 401).
- **"insufficient credits"** — top up at https://openrouter.ai/credits.
- **A model is not found (404)** — the slug in `backend/llm.py` is not available on OpenRouter for your account; check https://openrouter.ai/models.
- **Screenshot preview unavailable** — run `poetry run playwright install chromium` (Linux: `--with-deps`) or set `S2C_CHROMIUM_EXECUTABLE`.

To exercise the app without any key set `S2C_FAKE_MODEL=1`.
