/**
 * agent.ts — Agent wrapper for asset extraction.
 *
 * Provides buildAgent() and runExtractionWithAgent().
 * Uses createReactAgent from `@langchain/langgraph/prebuilt` with `extractAssets` tool.
 */

import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { ChatOpenRouter } from "@langchain/openrouter";
import {
  OPENROUTER_API_KEY,
  OPENROUTER_APP_TITLE,
  OPENROUTER_APP_URL,
  OPENROUTER_BASE_URL,
  ASSET_EXTRACTION_MODEL,
} from "./config.js";
import { extractAssets } from "./tool.js";
import type { ExtractionResult, ImageSource } from "./types.js";

const SYSTEM_PROMPT = `\
You are an asset extraction assistant.

You have one tool: \`extract_assets\`.

When the user provides images and labels, call \`extract_assets\` with those inputs
and return the result exactly as the tool returns it.
`;

/**
 * Builds the Agent with extractAssets tool.
 */
export function buildAgent(apiKey?: string) {
  const effectiveKey = apiKey ?? OPENROUTER_API_KEY;
  if (!effectiveKey) {
    throw new Error(
      "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill in your key."
    );
  }

  const model = new ChatOpenRouter({
    model: ASSET_EXTRACTION_MODEL,
    apiKey: effectiveKey,
    baseURL: OPENROUTER_BASE_URL,
    siteUrl: OPENROUTER_APP_URL,
    siteName: OPENROUTER_APP_TITLE,
  });

  return createReactAgent({
    llm: model,
    tools: [extractAssets],
    stateModifier: SYSTEM_PROMPT,
  });
}

/**
 * Public function to run extraction using Agent harness.
 */
export async function runExtractionWithAgent(
  images: ImageSource[],
  labels: string[],
  apiKey?: string
): Promise<ExtractionResult> {
  const effectiveKey = apiKey ?? OPENROUTER_API_KEY;
  return extractAssets.invoke({ images, labels, apiKey: effectiveKey });
}
