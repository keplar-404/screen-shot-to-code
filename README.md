# screenshot-to-code

Convert screenshots, mockups, Figma designs, and screen recordings into clean, functional code using AI. The easiest way to try this is using <a href="https://screenshottocode.com/?utm_source=github&utm_medium=readme&utm_campaign=oss_readme&utm_content=top_cta" target="_blank" rel="noopener noreferrer">the official, hosted product at screenshottocode.com →</a>


https://github.com/user-attachments/assets/ec08a5e6-9606-41c5-b03a-1bf47dfeba75


Supported stacks:

- HTML + Tailwind
- HTML + CSS
- React + Tailwind
- Vue + Tailwind
- Bootstrap
- Ionic + Tailwind

This fork runs the backend on **LangChain Deep Agents** with **OpenRouter** as the single LLM gateway: one API key gives you the OpenAI, Anthropic and Gemini models below (see `backend/ARCHITECTURE.md`).

Default AI models (all via OpenRouter):

- Gemini 3 Flash Preview and Gemini 3.1 Pro Preview
- GPT-5.5 and GPT-5.6 (sol / terra)
- Claude Opus 5, Claude Opus 4.8, Claude Fable 5, Claude Sonnet 4.6
- Image generation / editing through OpenRouter's Images API (default `google/gemini-2.5-flash-image`)

See the [Examples](#-examples) section below for more demos.

Screenshot to Code also supports taking a screen recording of a website in action and turning that into a functional prototype.

![google in app quick 3](https://github.com/abi/screenshot-to-code/assets/23818/8758ffa4-9483-4b9b-bb66-abd6d1594c33)

## 🛠 Getting Started

Choose the path that fits what you want to do:

- **Run locally:** best if you want to customize, self-host, or contribute.
- **Use the hosted app:** the fastest way to try Screenshot to Code with no local setup. <a href="https://screenshottocode.com/?utm_source=github&utm_medium=readme&utm_campaign=oss_readme&utm_content=getting_started_cta" target="_blank" rel="noopener noreferrer">Open the hosted app →</a>

Running locally requires API keys and a backend/frontend setup. The app has a React/Vite frontend and a FastAPI backend.

### API keys

You need **one** key: an [OpenRouter API key](https://openrouter.ai/keys). It unlocks every code-generation model, asset extraction (Gemini bounding boxes) and image generation/editing (OpenRouter Images API).

| Key | Required? | What it unlocks |
|-----|-----------|-----------------|
| `OPENROUTER_API_KEY` | **Yes** | All GPT / Claude / Gemini code-gen variants, video mode (Gemini), `extract_assets`, `generate_images`, `edit_images` |
| `REPLICATE_API_KEY` | Optional | `remove_backgrounds` only (OpenRouter has no background-removal model) |

Want to try the app with no key at all? Set `S2C_FAKE_MODEL=1` in `backend/.env` and a scripted local model produces a demo page through the full pipeline.

Run the backend (Python 3.11+; Poetry for package management — `pip install --upgrade poetry` if you don't have it):

```bash
cd backend
cp .env.example .env            # then put your OpenRouter key in .env
poetry install
# Install the Chromium browser used by the screenshot preview tool.
# On Linux, use `poetry run playwright install --with-deps chromium` to also
# install the required system libraries (needs sudo/apt).
poetry run playwright install chromium
poetry run uvicorn main:app --reload --port 7001
```

You can also enter the OpenRouter key (and an optional OpenRouter base URL / Replicate key) in the settings dialog in the frontend (gear icon). The Settings dialog also shows whether **screenshot preview** is available on your backend.

> **Screenshot preview** (optional) lets the agent render its own generated page in a headless browser and visually check its work. It's enabled automatically once Chromium is installed (the `playwright install chromium` step above, or automatically in the Docker image). If Chromium is missing, the app just skips the tool — the Settings dialog shows whether it's available. To use a specific Chromium binary set `S2C_CHROMIUM_EXECUTABLE=/path/to/chrome`.

Run the frontend:

```bash
cd frontend
pnpm install
pnpm dev
```

Open http://localhost:5173 to use the app.

If you prefer to run the backend on a different port, update `VITE_WS_BACKEND_URL` in `frontend/.env.local`.

## Docker

If you have Docker installed, run this from the root directory:

```bash
echo "OPENROUTER_API_KEY=sk-or-your-key" > .env
docker-compose up -d --build
```

The app will be up and running at http://localhost:5173. Note that you can't develop the application with this setup, as file changes won't trigger a rebuild.

## 🙋‍♂️ FAQs

- **I'm running into an error when setting up the backend. How can I fix it?** [Try this](https://github.com/abi/screenshot-to-code/issues/3#issuecomment-1814777959). If that still doesn't work, open an issue.
- **How do I get an OpenRouter API key?** Create one at https://openrouter.ai/keys and add credits to your OpenRouter account.
- **How can I configure a proxy for OpenRouter?** Set `OPENROUTER_BASE_URL` in `backend/.env` or the "OpenRouter Base URL" field in the settings dialog. Make sure the URL has `v1` in the path, for example: `https://xxx.xxxxx.xxx/api/v1`.
- **Can I pick a cheaper set of models?** Set `S2C_MODEL_SET=gemini` (or `anthropic` / `openai`) in `backend/.env`; the full model table is in `backend/llm.py`.
- **How can I update the backend host that my frontend connects to?** Configure `VITE_HTTP_BACKEND_URL` and `VITE_WS_BACKEND_URL` in `frontend/.env.local`. For example, set `VITE_HTTP_BACKEND_URL=http://124.10.20.1:7001`.
- **Seeing UTF-8 errors when running the backend?** On Windows, open the `.env` file with Notepad++, then go to Encoding and select UTF-8.
- **How can I provide feedback?** For feedback, feature requests, and bug reports, open an issue or ping me on [Twitter](https://twitter.com/_abi_).

## 📚 Examples

**NYTimes**

| Original                                                                                                                                                        | Replica                                                                                                                                                         |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img width="1238" alt="Screenshot 2023-11-20 at 12 54 03 PM" src="https://github.com/user-attachments/assets/6b0ae86c-1b0f-4598-a578-c7b62205b3e2"> | <img width="1435" height="737" alt="Screenshot 2026-06-15 at 3 06 37 PM" src="https://github.com/user-attachments/assets/48f0ab94-5fdc-41e7-ad6e-b4ad7ef69ae1" /> |


**Instagram**

https://github.com/user-attachments/assets/a335a105-f9cc-40e6-ac6b-64e5390bfc21

**Hacker News**


https://github.com/user-attachments/assets/205cb5c7-9c3c-438d-acd4-26dfe6e077e5
