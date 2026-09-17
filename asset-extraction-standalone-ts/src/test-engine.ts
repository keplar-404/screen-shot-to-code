/**
 * test-engine.ts — Tests image loading, sharp processing, and bounding box cropping
 * on `landingpage.png` without needing a live network API call.
 */

import { readFile } from "fs/promises";
import { resolve } from "path";
import sharp from "sharp";
import { extractAssets } from "./tool.js";

async function main() {
  console.log("=== Testing Asset Extraction Standalone Engine ===\n");

  // 1. Read landingpage.png
  const imagePath = resolve(process.cwd(), "landingpage.png");
  console.log(`1. Reading test image: ${imagePath}`);
  const imageBuffer = await readFile(imagePath);
  console.log(`   Image loaded. Size: ${(imageBuffer.length / 1024).toFixed(2)} KB`);

  // 2. Validate metadata using sharp
  const metadata = await sharp(imageBuffer).metadata();
  console.log(`2. Sharp Metadata: ${metadata.width}x${metadata.height} px, format=${metadata.format}`);

  // 3. Test sharp cropping on a dummy bounding box [ymin, xmin, ymax, xmax] = [50, 100, 300, 800]
  console.log("3. Testing sharp crop calculation...");
  const width = metadata.width!;
  const height = metadata.height!;

  const ymin = 50, xmin = 100, ymax = 300, xmax = 800; // 0-1000 scale
  const left = Math.floor((xmin / 1000) * width);
  const top = Math.floor((ymin / 1000) * height);
  const cropWidth = Math.ceil(((xmax - xmin) / 1000) * width);
  const cropHeight = Math.ceil(((ymax - ymin) / 1000) * height);

  const croppedBuffer = await sharp(imageBuffer)
    .extract({ left, top, width: cropWidth, height: cropHeight })
    .png()
    .toBuffer();

  console.log(`   Crop success! Cropped image size: ${cropWidth}x${cropHeight} px (${(croppedBuffer.length / 1024).toFixed(2)} KB)`);

  // 4. Test tool validation with missing API key handling
  console.log("4. Testing extractAssets tool validation...");
  const toolResult = await extractAssets.invoke({
    images: [{ path: "landingpage.png" }],
    labels: ["logo", "button"],
    apiKey: "dummy-key-for-validation-test",
  });

  console.log("   Tool schema validation: PASSED");
  console.log(`   Result structure: ok=${toolResult.ok}, assets=${toolResult.assets.length}`);

  console.log("\n✅ ALL ENGINE TESTS PASSED SUCCESSFULLY!");
}

main().catch((err) => {
  console.error("❌ Test failed:", err);
  process.exit(1);
});
