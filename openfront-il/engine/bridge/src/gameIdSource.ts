/**
 * Discovers candidate gameIds to fetch, from one of three sources:
 * a seed file (one gameId per line), CLI arguments, or the public
 * `/public/games` archive-listing endpoint (paginated by `Content-Range`).
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { parse as parseYaml } from "yaml";
import { z } from "zod";

const API_BASE = process.env.OPENFRONT_API_BASE ?? "https://api.openfront.io";

const GameListingItemSchema = z.object({
  game: z.string(),
  start: z.string(),
  end: z.string(),
  type: z.string(),
  mode: z.string(),
  difficulty: z.string(),
  numPlayers: z.number(),
  maxPlayers: z.number().nullable(),
  lobbyFillTime: z.number(),
  playerTeams: z.string().nullable(),
  rankedType: z.string(),
});

export const DataConfigSchema = z.object({
  maps: z.array(z.string()).optional(),
  startDate: z.string(),
  endDate: z.string(),
  gameType: z.enum(["Public", "Private", "Singleplayer"]).default("Public"),
  targetGameCount: z.number().int().positive().default(100),
});

export type DataConfig = z.infer<typeof DataConfigSchema>;

/** Loads and validates configs/data.yaml. */
export function loadDataConfig(configPath: string): DataConfig {
  const raw = fs.readFileSync(configPath, "utf8");
  return DataConfigSchema.parse(parseYaml(raw));
}

/** Reads a seed file of gameIds, one per line. Blank lines and lines
 * starting with `#` are ignored. */
export function fromSeedFile(seedFilePath: string): string[] {
  const text = fs.readFileSync(seedFilePath, "utf8");
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith("#"));
}

/** Extracts gameIds from CLI arguments (everything that isn't a `--flag`). */
export function fromCliArgs(argv: string[]): string[] {
  return argv.filter((arg) => !arg.startsWith("-"));
}

function parseContentRange(header: string | null): { end: number; total: number } | null {
  if (header === null) return null;
  const match = /^games (\d+)-(\d+)\/(\d+)$/.exec(header);
  if (match === null) return null;
  return { end: Number(match[2]), total: Number(match[3]) };
}

/**
 * Pages through `GET /public/games` for the configured date range and game
 * type, stopping once `config.targetGameCount` ids have been collected or
 * the listing is exhausted. The endpoint caps any single request to a
 * 2-day window and 1000 results, so a wide date range is paged in daily
 * chunks by the caller-provided `start`/`end` in `config`, and within a
 * chunk by `offset`/`limit`.
 */
export async function fromArchiveListing(config: DataConfig): Promise<string[]> {
  const ids: string[] = [];
  const limit = 1000;
  let offset = 0;

  while (ids.length < config.targetGameCount) {
    const url = new URL("/public/games", API_BASE);
    url.searchParams.set("start", config.startDate);
    url.searchParams.set("end", config.endDate);
    url.searchParams.set("type", config.gameType);
    url.searchParams.set("limit", String(limit));
    url.searchParams.set("offset", String(offset));

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`archive listing request failed: ${response.status} ${response.statusText}`);
    }
    const body: unknown = await response.json();
    const page = z.array(GameListingItemSchema).parse(body);
    for (const item of page) {
      ids.push(item.game);
      if (ids.length >= config.targetGameCount) break;
    }

    const range = parseContentRange(response.headers.get("Content-Range"));
    if (page.length === 0 || range === null || range.end >= range.total) {
      break;
    }
    offset += limit;
  }

  return ids;
}

/** Resolves the gameIds to fetch for this run, per CLI usage:
 *   tsx fetchGame.ts <gameId...>            explicit ids
 *   tsx fetchGame.ts --seed-file <path>     ids from a seed file
 *   tsx fetchGame.ts --config <path>        ids from the archive listing
 * Falls back to configs/data.yaml if no source is given. */
export async function resolveGameIds(argv: string[]): Promise<string[]> {
  const seedFlagIndex = argv.indexOf("--seed-file");
  if (seedFlagIndex >= 0) {
    const seedPath = argv[seedFlagIndex + 1];
    if (seedPath === undefined) {
      throw new Error("--seed-file requires a path argument");
    }
    return fromSeedFile(seedPath);
  }

  const positional = fromCliArgs(argv.filter((_, i) => argv[i - 1] !== "--config"));
  if (positional.length > 0) {
    return positional;
  }

  const configFlagIndex = argv.indexOf("--config");
  const explicitConfigPath = configFlagIndex >= 0 ? argv[configFlagIndex + 1] : undefined;
  const configPath =
    explicitConfigPath ?? path.join(import.meta.dirname, "..", "..", "..", "configs", "data.yaml");
  const config = loadDataConfig(configPath);
  return fromArchiveListing(config);
}
