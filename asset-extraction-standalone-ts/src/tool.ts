/**
 * tool.ts — LangChain @tool wrapping the extraction engine.
 *
 * Accepts:
 *   images  – one or more image sources (path | dataUrl | url)
 *   labels  – one or more asset description strings
 *
 * All labels are searched across all images in a single Gemini call.
 */

import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { readFile } from "fs/promises";
import { resolve } from "path";
import { OPENROUTER_API_KEY } from "./config.js";
import { extractAssetsFromImages } from "./extraction.js";
import type { ExtractionResult } from "./types.js";

// ---------------------------------------------------------------------------
// Input schema (Zod — equivalent of Python Pydantic)
// ---------------------------------------------------------------------------

const ImageSourceSchema = z.object({
  path: z
    .string()
    .optional()
    .describe("Local filesystem path (absolute or relative to cwd)."),
  dataUrl: z
    .string()
    .optional()
    .describe('Base64 data-URL, e.g. "data:image/png;base64,..."'),
  url: z
    .string()
    .url()
    .optional()
    .describe("Publicly reachable http/https URL — fetched automatically."),
});

const ExtractAssetsInputSchema = z.object({
  images: z
    .array(ImageSourceSchema)
    .min(1)
    .describe(
      "One or more source images to search. All labels are searched across all images; " +
        "Gemini picks the best match for each label."
    ),
  labels: z
    .array(z.string())
    .min(1)
    .describe(
      'One or more asset descriptions to locate. e.g. ["company logo top-left", "hero background"]'
    ),
  apiKey: z
    .string()
    .optional()
    .describe("OpenRouter API key. Falls back to OPENROUTER_API_KEY env var."),
});

// ---------------------------------------------------------------------------
// Image source resolvers
// ---------------------------------------------------------------------------

async function resolveImageSource(
  image: z.infer<typeof ImageSourceSchema>,
  index: number
): Promise<string> {
  const sources = [image.path, image.dataUrl, image.url].filter(Boolean);
  if (sources.length === 0) {
    throw new Error(`images[${index}] must provide one of: path, dataUrl, url`);
  }
  if (sources.length > 1) {
    throw new Error(`images[${index}] must provide exactly ONE of: path, dataUrl, url`);
  }

  // Local file path
  if (image.path) {
    const absPath = resolve(process.cwd(), image.path);
    const buf = await readFile(absPath);
    const ext = image.path.split(".").pop()?.toLowerCase() ?? "png";
    const mimeMap: Record<string, string> = {
      png: "image/png",
      jpg: "image/jpeg",
      jpeg: "image/jpeg",
      webp: "image/webp",
      heic: "image/heic",
      heif: "image/heif",
    };
    const mime = mimeMap[ext] ?? "image/png";
    return `data:${mime};base64,${buf.toString("base64")}`;
  }

  // Already a data-URL
  if (image.dataUrl) {
    return image.dataUrl;
  }

  // HTTP/HTTPS URL — Bun's native fetch
  const response = await fetch(image.url!);
  if (!response.ok) {
    throw new Error(`Failed to fetch image from ${image.url}: ${response.statusText}`);
  }
  const contentType =
    response.headers.get("content-type")?.split(";")[0]?.trim() ?? "image/png";
  const arrayBuf = await response.arrayBuffer();
  const base64 = Buffer.from(arrayBuf).toString("base64");
  return `data:${contentType};base64,${base64}`;
}

// ---------------------------------------------------------------------------
// The LangChain tool
// ---------------------------------------------------------------------------

export const extractAssets = tool(
  async ({
    images,
    labels,
    apiKey,
  }: z.infer<typeof ExtractAssetsInputSchema>): Promise<ExtractionResult> => {
    const resolvedKey = apiKey ?? OPENROUTER_API_KEY;
    if (!resolvedKey) {
      return {
        ok: false,
        error:
          "No OpenRouter API key found. Set OPENROUTER_API_KEY or pass apiKey.",
        assets: [],
      };
    }

    // Resolve all image sources concurrently
    let dataUrls: string[];
    try {
      dataUrls = await Promise.all(
        images.map((img, i) => resolveImageSource(img, i))
      );
    } catch (err) {
      return {
        ok: false,
        error: String(err),
        assets: [],
      };
    }

    // Clean labels
    const cleanLabels = labels.map((l) => l.trim()).filter(Boolean);
    if (cleanLabels.length === 0) {
      return { ok: false, error: "No valid labels provided.", assets: [] };
    }

    try {
      const result = await extractAssetsFromImages(dataUrls, cleanLabels, resolvedKey);
      return {
        ok: !result.error,
        assets: result.assets,
        ...(result.error ? { error: result.error } : {}),
      };
    } catch (err) {
      return {
        ok: false,
        error: `Extraction failed: ${err}`,
        assets: [],
      };
    }
  },
  {
    name: "extract_assets",
    description:
      "Locate and crop visual assets from one or more images. " +
      "Supply source images and a list of labels describing what to find. " +
      "All labels are searched across all images in a single Gemini call. " +
      "Returns crops as base64 PNG data-URLs with bounding boxes.",
    schema: ExtractAssetsInputSchema,
  }
);
