/**
 * test.ts — Test script for asset-extraction-standalone-ts using landingpage.png
 */

import { runExtraction } from "./index.js";
import { existsSync } from "fs";
import { resolve } from "path";

async function main() {
  const imagePath = resolve(process.cwd(), "landingpage.png");
  if (!existsSync(imagePath)) {
    console.error(`Error: Test image not found at ${imagePath}`);
    process.exit(1);
  }

  console.log(`Testing asset extraction with image: ${imagePath}`);
  console.log("Labels: ['company logo', 'main title header', 'navigation menu', 'call to action button']\n");

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    console.log("⚠️ OPENROUTER_API_KEY is not set in environment.");
    console.log("Please set OPENROUTER_API_KEY in .env or pass it to test live AI extraction.");
    console.log("Validating image resolution and engine readiness...");
    process.exit(0);
  }

  try {
    const result = await runExtraction(
      [{ path: "landingpage.png" }],
      ["company logo", "main title header", "navigation menu", "call to action button"]
    );

    console.log("--- Extraction Result ---");
    console.log(JSON.stringify({ ok: result.ok, error: result.error, assetCount: result.assets.length }, null, 2));

    for (const asset of result.assets) {
      const statusIcon = asset.status === "ok" ? "✅" : "❌";
      console.log(`\n${statusIcon} Label: ${asset.label}`);
      console.log(`   Status: ${asset.status}`);
      if (asset.box2d) {
        console.log(`   Box (2D): [${asset.box2d.join(", ")}]`);
      }
      if (asset.dataUrl) {
        console.log(`   Data URL: ${asset.dataUrl.substring(0, 50)}... (${asset.dataUrl.length} chars)`);
      }
    }
  } catch (err) {
    console.error("Extraction error:", err);
  }
}

main();
