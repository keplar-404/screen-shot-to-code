# Asset Extraction Standalone — TypeScript / Bun

TypeScript port of the Python asset extraction function.  
Uses **Bun** as runtime, **LangChain Deep Agent tool**, and **OpenRouter**.  
Pure function — no API server, no HTTP endpoints.

---

## Project layout

```
asset-extraction-standalone-ts/
├── src/
│   ├── index.ts      ← Public entry point  runExtraction()
│   ├── agent.ts      ← Deep Agent wrapper  buildAgent()
│   ├── tool.ts       ← LangChain @tool  extract_assets
│   ├── extraction.ts ← Core bounding-box engine (sharp + OpenRouter)
│   ├── config.ts     ← Env-var settings
│   └── types.ts      ← TypeScript interfaces
├── package.json
├── tsconfig.json
└── .env.example
```

---

## Quick start

```bash
cd asset-extraction-standalone-ts

# Install deps
bun install

# Set your API key
cp .env.example .env
# → edit .env and set OPENROUTER_API_KEY=sk-or-v1-...
```

---

## Usage

### From code

```typescript
import { runExtraction } from "./src/index.ts";

// One image, multiple labels
const result = await runExtraction(
  [{ path: "screenshot.png" }],
  ["company logo", "hero background photo", "nav icon", "CTA button"]
);

// Multiple images, multiple labels
const result = await runExtraction(
  [{ path: "home.png" }, { url: "https://example.com/shot.jpg" }],
  ["company logo", "team photo"]
);

for (const asset of result.assets) {
  console.log(asset.label, "→", asset.status);
  // asset.dataUrl is a "data:image/png;base64,…" string when status === "ok"
}
```

### From the CLI

```bash
# One image, multiple labels
bun run src/index.ts --images screenshot.png --labels "company logo" "hero photo" "nav icon"

# Multiple images, multiple labels
bun run src/index.ts --images home.png about.png --labels "logo" "team photo"
```

### Embed the tool in your own Deep Agent

```typescript
import { createDeepAgent } from "deepagents";
import { ChatOpenRouter } from "@langchain/openrouter";
import { extractAssets } from "./src/tool.ts";

const agent = await createDeepAgent({
  model: new ChatOpenRouter({ model: "google/gemini-3.6-flash" }),
  tools: [extractAssets],   // ← drop it in
  systemPrompt: "You are an asset extraction assistant.",
});
```

---

## Image source options (pick exactly one)

| Field | Description |
|-------|-------------|
| `path` | Local file path (absolute or relative) |
| `dataUrl` | `"data:image/png;base64,…"` string |
| `url` | Public `http(s)://` URL — fetched automatically |

---

## Return value

```typescript
{
  ok: boolean,
  assets: [
    {
      label:      "company logo",
      dataUrl:    "data:image/png;base64,…",  // PNG crop or null
      status:     "ok",                        // "ok" | "missing" | "error"
      box2d:      [12, 18, 80, 210],           // ymin xmin ymax xmax (0-1000)
      imageIndex: 1,                           // which image (1-based)
    }
  ],
  error?: "…"   // only when ok=false
}
```

---

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | ✅ | — | Your OpenRouter key |
| `ASSET_EXTRACTION_MODEL` | — | `google/gemini-3.6-flash` | Detection model |
| `OPENROUTER_BASE_URL` | — | `https://openrouter.ai/api/v1` | Override for proxies |

> **Bun auto-loads `.env`** — no `dotenv` import needed.
