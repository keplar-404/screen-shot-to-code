# Backend architecture (LangChain Deep Agents + OpenRouter)

The backend is a FastAPI service. Code generation runs on **LangChain Deep
Agents** (`deepagents` 0.7.x) and every LLM call goes through **OpenRouter**
with one API key. The WebSocket protocol the frontend speaks is unchanged from
the original project.

```
frontend ──ws /generate-code──▶ routes/generate_code.py   (pipeline: params → status → prompt → generation)
                                        │  one DeepAgentRunner per variant (asyncio.gather)
                                        ▼
                               agent/deep_runner.py
                                 ├─ llm.py + models/factory.py      Llm → OpenRouter slug + reasoning.effort → ChatOpenRouter
                                 ├─ prompts/to_langchain.py         prompt dicts → LangChain messages (video → video_url)
                                 ├─ agent/tools/*                   create_file, edit_file, generate_images, remove_backgrounds,
                                 │                                  edit_images, extract_assets, screenshot_preview,
                                 │                                  save_assets, retrieve_option
                                 ├─ agent/middleware/*              StepLimit, Budget, ToolStartEvents, AnthropicImageLimits, RunRecorder
                                 ├─ agent/profiles.py               deepagents harness profile for "openrouter" models
                                 └─ create_deep_agent(...)          LangGraph agent loop
                                        │ astream(stream_mode=[messages, updates, custom, values])
                                        ▼
                               agent/stream_bridge.py  →  thinking / assistant / toolStart / setCode / toolResult
```

## Request flow

1. `ParameterExtractionStage` validates the request, resolves `openRouterApiKey`
   (settings dialog value wins over `OPENROUTER_API_KEY`), stages uploaded
   images as temporary assets and parses the prompt/history.
2. `prompts/pipeline.py` builds the provider-neutral prompt (unchanged:
   create-from-image/text/video, update-from-history, update-from-file-snapshot).
3. `ModelSelectionStage` picks the variant models (`routes/model_choice_sets.py`).
   Video always uses the Gemini variants; `S2C_MODEL_SET` can narrow the set.
4. `DeepAgentRunner.run(model, prompt_messages)`:
   - seeds the working file for updates (`seed_file_state_from_messages`);
   - decides which tools to offer (`agent/tools/registry.py`, same gating rules
     as before: `extract_assets` only with a still image, `screenshot_preview`
     only when Chromium works, `generate_images` only when enabled);
   - builds `ChatOpenRouter` for the model (`models/factory.py`) with
     `reasoning={"effort": ...}`, `max_tokens=50000`, `usage: {include: true}`
     and `provider: {require_parameters: true, ignore: ["azure"]}`;
   - creates the deep agent with a custom state (`S2CState`: `html_file`,
     `model_calls`, `spent_usd`) and context (`S2CContext`: per-variant config
     and the mutable working file the file tools share);
   - streams the run and lets `StreamBridge` translate it to WebSocket messages;
   - returns the final HTML (or raises `EmptyOutputError`).

## Deep Agents configuration (why these settings)

- `agent/profiles.py` registers a `HarnessProfile` for provider `openrouter`
  that removes the `task` subagent tool, hides the built-in `read_file` and
  drops `SummarizationMiddleware`. The app writes code exclusively through
  `create_file`/`edit_file`, and summarisation would compact the input
  screenshot out of context on long runs.
- `FilesystemMiddleware(tools=["read_file"], tool_token_limit_before_evict=None,
  human_message_token_limit_before_evict=None)` keeps deep agents' message
  eviction off: this app legitimately sends very large human messages (the
  current file in update mode) and tool results (`retrieve_option`).
- `StepLimitMiddleware` (30 model calls) and `BudgetMiddleware`
  (`GENERATION_MAX_COST_USD`, read from OpenRouter's reported `cost`) reproduce
  the original loop's limits. The budget aborts only when the run would
  continue (the last turn requested tools).
- `ToolStartEventsMiddleware` emits `toolStart` when a tool actually starts
  executing (after the budget check), matching the original event order.
- `AnthropicImageLimitsMiddleware` resizes base64 images to Claude's limits
  (5 MB / 8000 px, 2000 px when a request carries >20 images) for Anthropic
  variants — OpenRouter forwards image bytes unchanged.
- Tools return `Command(update=...)`: a `ToolMessage` (with image parts in
  OpenRouter `image_url` format for screenshots / crops / generated images),
  the mirrored `html_file` state, and custom stream events via
  `runtime.stream_writer` (`toolResult`, `setCode`).

## Streaming → WebSocket protocol

| LangGraph stream event | WebSocket message |
|---|---|
| `messages`: `AIMessageChunk.additional_kwargs.reasoning_content` | `thinking` |
| `messages`: text content | `assistant` |
| `messages`: `tool_call_chunks` for `create_file` | `toolStart` once, then throttled `setCode` from the partial JSON `content` |
| `custom`: `toolStart` (from middleware) | `toolStart` (+ cosmetic code preview stream for `create_file`) |
| `custom`: `toolResult` / `setCode` (from tools) | `toolResult`, `setCode` |
| `updates` (`tools` node) for a call that never emitted a result | `toolResult` with `ok: false` |
| end of run | final `setCode` + `variantComplete` (route) |

## External services

| Service | Used for | Key |
|---|---|---|
| OpenRouter chat completions | all code-generation models, asset-extraction detector | `OPENROUTER_API_KEY` |
| OpenRouter Images API (`/api/v1/images`) | `generate_images`, `edit_images` | `OPENROUTER_API_KEY` |
| Replicate (optional) | `remove_backgrounds` | `REPLICATE_API_KEY` |
| ScreenshotOne (optional, unchanged) | `/api/screenshot` URL capture | entered in the UI |
| Local Chromium via Playwright | `screenshot_preview` tool | none |

## Running without a key

`S2C_FAKE_MODEL=1` swaps `ChatOpenRouter` for `models/fake.py`, a scripted
model that streams a `create_file` (or `edit_file` for updates), calls
`screenshot_preview` when Chromium is available and finishes with a summary.
Every route, the WebSocket protocol and the frontend work end-to-end in this
mode.

## Tests

`poetry run pytest` — 191 tests, including a scripted-model harness
(`tests/scripted_model.py`) that exercises the full deep-agent loop and the
WebSocket protocol without network access. `poetry run pyright` — 0 errors.
