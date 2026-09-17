/**
 * Config — all values from environment variables.
 * Bun automatically loads .env — no dotenv import needed.
 */

export const OPENROUTER_API_KEY: string | undefined =
  process.env.OPENROUTER_API_KEY;

export const OPENROUTER_BASE_URL: string =
  process.env.OPENROUTER_BASE_URL ?? "https://openrouter.ai/api/v1";

export const OPENROUTER_APP_URL: string =
  process.env.OPENROUTER_APP_URL ??
  "https://github.com/abi/screenshot-to-code";

export const OPENROUTER_APP_TITLE: string =
  process.env.OPENROUTER_APP_TITLE ?? "asset-extraction-standalone-ts";

/** Bounding-box detection model — reached via OpenRouter */
export const ASSET_EXTRACTION_MODEL: string =
  process.env.ASSET_EXTRACTION_MODEL ?? "google/gemini-3.6-flash";
