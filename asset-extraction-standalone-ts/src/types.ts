/**
 * Types shared across the project.
 */

/** One image source — supply exactly one of path, dataUrl, or url. */
export interface ImageSource {
  /** Local filesystem path (absolute or relative to cwd). */
  path?: string;
  /** Base64 data-URL, e.g. "data:image/png;base64,...". */
  dataUrl?: string;
  /** Publicly reachable http/https URL — fetched automatically. */
  url?: string;
}

/** One extracted asset in the result. */
export interface ExtractedAsset {
  /** The label you supplied. */
  label: string;
  /** Base64 PNG data-URL of the cropped asset, or null if not found. */
  dataUrl: string | null;
  /** "ok" | "missing" | "error" */
  status: "ok" | "missing" | "error";
  /** [ymin, xmin, ymax, xmax] normalised 0-1000, or null. */
  box2d: [number, number, number, number] | null;
  /** Which input image the asset was found in (1-based), or null. */
  imageIndex: number | null;
}

/** Return value of runExtraction(). */
export interface ExtractionResult {
  ok: boolean;
  assets: ExtractedAsset[];
  error?: string;
}
