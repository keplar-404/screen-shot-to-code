/**
 * index.ts — Public entry point.
 *
 * runExtraction(images, labels, apiKey?)  → Promise<ExtractionResult>
 *
 * Examples:
 *
 *   // One image, multiple labels
 *   const result = await runExtraction(
 *     [{ path: "screenshot.png" }],
 *     ["company logo", "hero background photo", "nav icon"]
 *   );
 *
 *   // Multiple images, multiple labels
 *   const result = await runExtraction(
 *     [{ path: "home.png" }, { url: "https://example.com/shot.jpg" }],
 *     ["company logo", "team photo"]
 *   );
 *
 *   // Sync-style via top-level await (Bun supports this natively)
 *   console.log(JSON.stringify(result.assets, null, 2));
 *
 * CLI:
 *   bun run src/index.ts --images shot.png --labels "logo" "hero photo"
 */

import { existsSync, mkdirSync, writeFileSync } from "fs";
import { resolve } from "path";
import { extractAssets } from "./tool.js";
import { OPENROUTER_API_KEY } from "./config.js";
import type { ExtractionResult, ImageSource } from "./types.js";

// ---------------------------------------------------------------------------
// Public function
// ---------------------------------------------------------------------------

/**
 * Locate and crop visual assets from one or more images.
 *
 * @param images  - Array of image sources. Each must have exactly one of:
 *                  `path` (local file), `dataUrl` (base64), `url` (http/https)
 * @param labels  - Array of asset descriptions to find, e.g.
 *                  ["company logo", "hero background", "nav icon"]
 * @param apiKey  - Optional OpenRouter API key (falls back to env var)
 * @returns       ExtractionResult with `ok`, `assets[]`, and optional `error`
 */
export async function runExtraction(
  images: ImageSource[],
  labels: string[],
  apiKey?: string
): Promise<ExtractionResult> {
  return extractAssets.invoke({ images, labels, apiKey });
}

// Re-export types, tool, and agent builder
export { extractAssets } from "./tool.js";
export { buildAgent, runExtractionWithAgent } from "./agent.js";
export type { ExtractionResult, ImageSource, ExtractedAsset } from "./types.js";

// ---------------------------------------------------------------------------
// CLI entry point — only runs when executed directly
// ---------------------------------------------------------------------------

if (import.meta.main) {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes("--help")) {
    console.log(`
Asset Extraction Standalone (TypeScript / Bun)

Usage:
  bun run src/index.ts --images <path...> --labels <label...>

Options:
  --images   One or more local image file paths
  --labels   One or more asset description labels

Examples:
  # One image, multiple labels
  bun run src/index.ts --images screenshot.png --labels "company logo" "hero photo" "nav icon"

  # Multiple images, multiple labels
  bun run src/index.ts --images home.png about.png --labels "logo" "team photo"
`);
    process.exit(0);
  }

  // Parse --images and --labels
  const imagesIdx = args.indexOf("--images");
  const labelsIdx = args.indexOf("--labels");

  if (imagesIdx === -1 || labelsIdx === -1) {
    console.error("Error: --images and --labels are both required.");
    process.exit(1);
  }

  // Collect values until the next flag
  function collectArgs(arr: string[], startIdx: number): string[] {
    const values: string[] = [];
    for (let i = startIdx + 1; i < arr.length; i++) {
      if (arr[i]!.startsWith("--")) break;
      values.push(arr[i]!);
    }
    return values;
  }

  const imagePaths = collectArgs(args, imagesIdx);
  const labels = collectArgs(args, labelsIdx);

  if (imagePaths.length === 0) {
    console.error("Error: --images requires at least one file path.");
    process.exit(1);
  }
  if (labels.length === 0) {
    console.error("Error: --labels requires at least one label.");
    process.exit(1);
  }

  if (!OPENROUTER_API_KEY) {
    console.error(
      "Error: OPENROUTER_API_KEY is not set.\n" +
        "Copy .env.example to .env and fill in your key."
    );
    process.exit(1);
  }

  const images: ImageSource[] = imagePaths.map((p) => ({ path: p }));

  console.log(`\nExtracting ${labels.length} asset(s) from ${images.length} image(s)...\n`);

  const result = await runExtraction(images, labels);

  const outputDir = resolve(process.cwd(), "output");
  if (!existsSync(outputDir)) {
    mkdirSync(outputDir, { recursive: true });
  }

  console.log(`\n--- Extraction Results ---`);
  for (const asset of result.assets) {
    const icon = asset.status === "ok" ? "✓" : "✗";
    console.log(`\n${icon} ${asset.label} → ${asset.status}`);
    if (asset.box2d) {
      console.log(`  box: [${asset.box2d.join(", ")}]  image #${asset.imageIndex}`);
    }
    if (asset.dataUrl) {
      const filename = `${asset.label.replace(/[^a-z0-9]/gi, "_").toLowerCase()}.png`;
      const filePath = resolve(outputDir, filename);
      const base64Data = asset.dataUrl.split(",")[1]!;
      writeFileSync(filePath, Buffer.from(base64Data, "base64"));
      console.log(`  💾 Saved cropped image to: output/${filename}`);
    }
  }
}
