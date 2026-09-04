/**
 * Fetches finished games from the public OpenFront API and writes them,
 * verbatim, to data/raw/{gameId}.json. See engine/bridge/README.md for the
 * full contract: rate-limited, retried with backoff, never overwrites an
 * existing raw file.
 */
import * as fs from "node:fs";
import { pathToFileURL } from "node:url";
import { GameRecordSchema, RAW_DIR, rawGamePath } from "./types";
import { resolveGameIds } from "./gameIdSource";

const API_BASE = process.env.OPENFRONT_API_BASE ?? "https://api.openfront.io";
const RATE_LIMIT_MS = Number(process.env.OPENFRONT_FETCH_RATE_LIMIT_MS ?? 500);
const MAX_RETRIES = Number(process.env.OPENFRONT_FETCH_MAX_RETRIES ?? 5);

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithBackoff(url: string): Promise<Response> {
  let attempt = 0;
  for (;;) {
    const response = await fetch(url);
    if (response.ok || attempt >= MAX_RETRIES) {
      return response;
    }
    attempt++;
    const backoffMs = RATE_LIMIT_MS * 2 ** attempt;
    console.log(
      `[fetchGame] ${url} -> ${response.status}, retrying in ${backoffMs}ms (attempt ${attempt}/${MAX_RETRIES})`,
    );
    await sleep(backoffMs);
  }
}

/**
 * Fetches one game and writes it to data/raw/{gameId}.json.
 * Returns "written", "skipped-exists", or "failed".
 */
export async function fetchGame(gameId: string): Promise<"written" | "skipped-exists" | "failed"> {
  const destination = rawGamePath(gameId);
  if (fs.existsSync(destination)) {
    return "skipped-exists";
  }

  const url = `${API_BASE}/public/game/${gameId}`;
  const response = await fetchWithBackoff(url);
  if (!response.ok) {
    console.error(
      `[fetchGame] ${gameId}: request failed with ${response.status} ${response.statusText}`,
    );
    return "failed";
  }

  const body: unknown = await response.json();
  const parsed = GameRecordSchema.safeParse(body);
  if (!parsed.success) {
    console.error(
      `[fetchGame] ${gameId}: response did not match GameRecordSchema: ${parsed.error.message}`,
    );
    return "failed";
  }
  if (parsed.data.turns.length === 0) {
    console.error(`[fetchGame] ${gameId}: empty game (no turns), skipping`);
    return "failed";
  }

  fs.mkdirSync(RAW_DIR, { recursive: true });
  const temporaryPath = `${destination}.tmp`;
  fs.writeFileSync(temporaryPath, JSON.stringify(body));
  fs.renameSync(temporaryPath, destination);
  return "written";
}

async function main(): Promise<void> {
  const gameIds = await resolveGameIds(process.argv.slice(2));
  console.log(`[fetchGame] resolved ${gameIds.length} candidate gameId(s)`);

  let written = 0;
  let skipped = 0;
  let failed = 0;
  for (const gameId of gameIds) {
    const result = await fetchGame(gameId);
    if (result === "written") written++;
    else if (result === "skipped-exists") skipped++;
    else failed++;
    if (result === "written") {
      await sleep(RATE_LIMIT_MS);
    }
  }
  console.log(`[fetchGame] done: ${written} written, ${skipped} already present, ${failed} failed`);
}

const isMain =
  process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  main().catch((error: unknown) => {
    console.error(error);
    process.exit(1);
  });
}
