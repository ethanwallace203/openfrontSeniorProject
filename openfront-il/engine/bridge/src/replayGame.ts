/**
 * Boots the vendored engine headlessly and replays a raw game record
 * (data/raw/{gameId}.json) deterministically, tick by tick, writing
 * per-tick snapshots to data/snapshots/{gameId}.jsonl.
 *
 * Determinism is verified the same way the live clients did originally:
 * the engine emits a state hash every 10 ticks, and this replay compares
 * it against the hash the record captured during the real, live game. A
 * mismatch means the replay has desynced from the original and the game
 * is skipped rather than trusted.
 *
 * Scope limitation (documented, not accidental): this project pins a
 * single engine commit (see engine/README.md) and always replays against
 * it, unlike the upstream engine's own history of checking out the exact
 * commit each archived game was played on. A game recorded on a
 * meaningfully different engine commit will simply fail hash verification
 * here and be skipped — that failure is indistinguishable from a genuine
 * replay bug, which is an acceptable tradeoff for a prototype fetching
 * recently-played games against a recently-pinned engine commit, but
 * would need per-commit engine checkouts to fix properly at scale.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import {
  Game,
  GameMapSize,
  GameMapType,
  GameType,
  PlayerInfo,
  PlayerType,
  UnitType,
} from "../../openfront-ai/openfront/src/core/game/Game";
import { createGame } from "../../openfront-ai/openfront/src/core/game/GameImpl";
import { GameUpdateType } from "../../openfront-ai/openfront/src/core/game/GameUpdates";
import type {
  GameMapLoader,
  MapData,
} from "../../openfront-ai/openfront/src/core/game/GameMapLoader";
import { createNationsForGame } from "../../openfront-ai/openfront/src/core/game/NationCreation";
import type {
  TerrainMapData,
  MapManifest,
} from "../../openfront-ai/openfront/src/core/game/TerrainMapLoader";
import { genTerrainFromBin } from "../../openfront-ai/openfront/src/core/game/TerrainMapLoader";
import { Config } from "../../openfront-ai/openfront/src/core/configuration/Config";
import { Executor } from "../../openfront-ai/openfront/src/core/execution/ExecutionManager";
import { DoomsdayClockExecution } from "../../openfront-ai/openfront/src/core/execution/DoomsdayClockExecution";
import { RecomputeRailClusterExecution } from "../../openfront-ai/openfront/src/core/execution/RecomputeRailClusterExecution";
import { SpawnTimerExecution } from "../../openfront-ai/openfront/src/core/execution/SpawnTimerExecution";
import { WinCheckExecution } from "../../openfront-ai/openfront/src/core/execution/WinCheckExecution";
import { PseudoRandom } from "../../openfront-ai/openfront/src/core/PseudoRandom";
import { decompressGameRecord, simpleHash } from "../../openfront-ai/openfront/src/core/Util";
import type { GameRecord, GameStartInfo } from "../../openfront-ai/openfront/src/core/Schemas";
import { ENGINE_ROOT, RAW_DIR, rawGamePath, snapshotPath } from "./types";
import type { StampedIntent, TickSnapshot } from "./types";
import { buildTerrainHeader, buildTickSnapshot, SnapshotWriter } from "./snapshotWriter";

const DEFAULT_SNAPSHOT_EVERY = Number(process.env.OPENFRONT_SNAPSHOT_EVERY ?? 10);

// ---------------------------------------------------------------------------
// Node-side map loading. GameMapLoader's shipped implementations assume a
// browser (fetch/File APIs); this reads the same .bin/manifest.json files
// directly off disk from the vendored engine's own resources/maps/ tree.
// ---------------------------------------------------------------------------

function mapSlug(map: GameMapType): string {
  return map.toLowerCase().replace(/\s+/g, "");
}

function createNodeMapLoader(): GameMapLoader {
  return {
    getMapData(map: GameMapType): MapData {
      const dir = path.join(ENGINE_ROOT, "resources", "maps", mapSlug(map));
      const readBin = (fileName: string) => async (): Promise<Uint8Array> => {
        const buffer = await fs.promises.readFile(path.join(dir, fileName));
        return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
      };
      return {
        mapBin: readBin("map.bin"),
        map4xBin: readBin("map4x.bin"),
        map16xBin: readBin("map16x.bin"),
        manifest: async (): Promise<MapManifest> => {
          const text = await fs.promises.readFile(path.join(dir, "manifest.json"), "utf8");
          return JSON.parse(text) as MapManifest;
        },
        webpPath: path.join(dir, "thumbnail.webp"),
        // Client-rendering-only (decorative map layer PNGs, browser
        // ImageBitmap); a headless replay never calls this.
        layerPng: (): Promise<ImageBitmap> => {
          throw new Error(
            "layerPng is a client-rendering concern, not available in a headless replay",
          );
        },
      };
    },
  };
}

/**
 * Loads terrain WITHOUT the engine's own `loadTerrainMap()` module-level
 * cache. That cache returns the same GameMap object for repeat calls with
 * the same (map, size) key — fine for a single live game, but this
 * process replays many games back to back, and GameMap carries mutable
 * per-game ownership state. Reusing a cached instance across games would
 * silently leak one game's tile ownership into the next.
 */
async function loadFreshTerrain(map: GameMapType, mapSize: GameMapSize): Promise<TerrainMapData> {
  const mapFiles = createNodeMapLoader().getMapData(map);
  const manifest = await mapFiles.manifest();

  // Mirrors TerrainMapLoader.ts's loadTerrainMap(), minus its module-level
  // cache (see this function's own doc comment) and minus layer-image
  // loading (purely a client rendering concern this headless replay never
  // touches — layerPng() is stubbed to throw in createNodeMapLoader()).
  const gameMap =
    mapSize === GameMapSize.Normal
      ? await genTerrainFromBin(manifest.map, await mapFiles.mapBin())
      : await genTerrainFromBin(manifest.map4x, await mapFiles.map4xBin());
  const miniGameMap =
    mapSize === GameMapSize.Normal
      ? await genTerrainFromBin(manifest.map4x, await mapFiles.map4xBin())
      : await genTerrainFromBin(manifest.map16x, await mapFiles.map16xBin());

  if (mapSize === GameMapSize.Compact) {
    for (const nation of [...manifest.nations, ...(manifest.additionalNations ?? [])]) {
      if (nation.coordinates !== undefined) {
        nation.coordinates = [
          Math.floor(nation.coordinates[0] / 2),
          Math.floor(nation.coordinates[1] / 2),
        ];
      }
    }
  }

  let teamGameSpawnAreas = manifest.teamGameSpawnAreas;
  if (mapSize === GameMapSize.Compact && teamGameSpawnAreas) {
    const scaled: typeof teamGameSpawnAreas = {};
    for (const [key, areas] of Object.entries(teamGameSpawnAreas)) {
      scaled[key] = areas.map((a) => ({
        x: Math.floor(a.x / 2),
        y: Math.floor(a.y / 2),
        width: Math.max(1, Math.floor(a.width / 2)),
        height: Math.max(1, Math.floor(a.height / 2)),
      }));
    }
    teamGameSpawnAreas = scaled;
  }

  return {
    nations: manifest.nations,
    additionalNations: manifest.additionalNations ?? [],
    gameMap,
    miniGameMap,
    teamGameSpawnAreas,
  };
}

// ---------------------------------------------------------------------------
// Boot sequence — mirrors the engine's own GameRunner.init(), minus the
// client-facing view-data machinery GameRunner also carries (name-box
// placement, packed wire-update draining), which this headless replay has
// no use for.
// ---------------------------------------------------------------------------

async function bootGame(record: GameRecord): Promise<{ game: Game; executor: Executor }> {
  const info = record.info as GameStartInfo;
  const gameConfig = info.config;

  const mapType = gameConfig.gameMap as GameMapType;
  if (!Object.values(GameMapType).includes(mapType)) {
    throw new Error(`unknown map: ${gameConfig.gameMap}`);
  }

  // isReplay=false is deliberate: to reproduce the exact hashes the live
  // clients reported, the engine must be configured exactly as it was
  // during the original live game, which was not itself a replay.
  const config = new Config(gameConfig, null, false, info.listed ?? false);
  const terrain = await loadFreshTerrain(mapType, gameConfig.gameMapSize);
  const random = new PseudoRandom(simpleHash(info.gameID));

  // Humans consume random.nextID() in player order, then nations — this
  // order must match the live game exactly for player IDs to line up.
  const humans = info.players.map(
    (p) =>
      new PlayerInfo(
        p.username,
        PlayerType.Human,
        p.clientID,
        random.nextID(),
        p.isLobbyCreator ?? false,
        p.clanTag,
        p.friends ?? [],
        p.teamIndex ?? null,
      ),
  );
  const nations = createNationsForGame(
    info,
    terrain.nations,
    terrain.additionalNations,
    humans.length,
    random,
  );

  const game: Game = createGame(
    humans,
    nations,
    terrain.gameMap,
    terrain.miniGameMap,
    config,
    terrain.teamGameSpawnAreas,
  );
  // Purchased tribe names (info.tribes) shift how TribeSpawner's own PRNG
  // instance gets consumed — passing an empty array here when the live
  // game actually had purchased names desyncs every tribe's spawn tile and
  // ID from that point on. See TribeSpawner.spawnTribes()'s own comment:
  // "Guarded so games without purchased names consume the PRNG exactly as
  // before" — the guard only holds if this argument is passed correctly.
  const purchasedTribeNames = info.tribes?.map((t) => t.name);
  const executor = new Executor(game, info.gameID, undefined, purchasedTribeNames);

  if (gameConfig.gameType !== GameType.Singleplayer) {
    game.addExecution(new SpawnTimerExecution());
  }
  if (config.spawnNations()) {
    game.addExecution(...executor.nationExecutions());
  }
  if (config.isRandomSpawn()) {
    game.addExecution(...executor.spawnPlayers());
  }
  if (config.bots() > 0) {
    game.addExecution(...executor.spawnTribes(config.bots()));
  }
  game.addExecution(new WinCheckExecution());
  if (config.doomsdayClockConfig().enabled) {
    game.addExecution(new DoomsdayClockExecution());
  }
  if (!config.isUnitDisabled(UnitType.Factory)) {
    game.addExecution(new RecomputeRailClusterExecution(game.railNetwork()));
  }

  return { game, executor };
}

function normalizeWinner(raw: unknown): { kind: string; id: string } | null {
  if (!Array.isArray(raw) || raw.length < 2) return null;
  return { kind: String(raw[0]), id: String(raw[1]) };
}

export interface ReplayResult {
  ok: boolean;
  reason?: string;
  ticks: number;
  snapshotsWritten: number;
  hashesChecked: number;
}

export async function replayGame(record: GameRecord, snapshotEvery: number): Promise<ReplayResult> {
  const info = record.info as GameStartInfo;
  const humanClientIDs = new Set(info.players.map((p) => p.clientID));

  const { game, executor } = await bootGame(record);
  const decompressed = decompressGameRecord(record);

  const writer = new SnapshotWriter(info.gameID);
  writer.writeHeader(
    buildTerrainHeader(
      game,
      info.gameID,
      record.gitCommit,
      String(info.config.gameMap),
      snapshotEvery,
    ),
  );

  let pendingActions: Record<string, StampedIntent[]> = {};
  let winner: { kind: string; id: string } | null = null;
  let hashesChecked = 0;
  let snapshotsWritten = 0;

  try {
    for (const turn of decompressed.turns) {
      for (const intent of turn.intents) {
        if (humanClientIDs.has(intent.clientID)) {
          (pendingActions[intent.clientID] ??= []).push(intent);
        }
      }

      game.addExecution(...executor.createExecs(turn));
      const updates = game.executeNextTick();

      const hashUpdates = updates[GameUpdateType.Hash] as
        { tick: number; hash: number }[] | undefined;
      for (const hashUpdate of hashUpdates ?? []) {
        const archived = decompressed.turns[hashUpdate.tick]?.hash;
        if (archived === null || archived === undefined) continue;
        hashesChecked++;
        if (archived !== hashUpdate.hash) {
          return {
            ok: false,
            reason: `desync at tick ${hashUpdate.tick} (ours ${hashUpdate.hash} != archived ${archived})`,
            ticks: game.ticks(),
            snapshotsWritten,
            hashesChecked,
          };
        }
      }

      const winUpdates = updates[GameUpdateType.Win] as { winner: unknown }[] | undefined;
      const firstWinUpdate = winUpdates?.[0];
      if (firstWinUpdate !== undefined) {
        winner = normalizeWinner(firstWinUpdate.winner);
      }

      if (!game.inSpawnPhase() && game.ticks() % snapshotEvery === 0) {
        const tick: TickSnapshot = buildTickSnapshot(game, humanClientIDs, pendingActions, winner);
        writer.writeTick(tick);
        snapshotsWritten++;
        pendingActions = {};
      }
    }
  } finally {
    writer.close();
  }

  return { ok: true, ticks: game.ticks(), snapshotsWritten, hashesChecked };
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const everyFlagIndex = args.indexOf("--every");
  const snapshotEvery =
    everyFlagIndex >= 0 && args[everyFlagIndex + 1] !== undefined
      ? Number(args[everyFlagIndex + 1])
      : DEFAULT_SNAPSHOT_EVERY;
  const explicitIds = args.filter((arg) => !arg.startsWith("-") && arg !== String(snapshotEvery));

  const gameIds =
    explicitIds.length > 0
      ? explicitIds
      : fs
          .readdirSync(RAW_DIR)
          .filter((name) => name.endsWith(".json"))
          .map((name) => name.slice(0, -".json".length));

  let ok = 0;
  let skipped = 0;
  let failed = 0;
  for (const gameId of gameIds) {
    if (fs.existsSync(snapshotPath(gameId))) {
      skipped++;
      continue;
    }
    const record = JSON.parse(fs.readFileSync(rawGamePath(gameId), "utf8")) as GameRecord;
    try {
      const result = await replayGame(record, snapshotEvery);
      if (result.ok) {
        ok++;
        console.log(
          `[replayGame] ${gameId}: ok, ${result.ticks} ticks, ${result.snapshotsWritten} snapshots, ${result.hashesChecked} hashes verified`,
        );
      } else {
        failed++;
        fs.rmSync(snapshotPath(gameId), { force: true });
        console.error(`[replayGame] ${gameId}: FAILED — ${result.reason}`);
      }
    } catch (error: unknown) {
      failed++;
      fs.rmSync(snapshotPath(gameId), { force: true });
      console.error(
        `[replayGame] ${gameId}: crashed —`,
        error instanceof Error ? error.stack : error,
      );
    }
  }
  console.log(
    `[replayGame] done: ${ok} ok, ${skipped} already replayed, ${failed} failed, ${gameIds.length} total`,
  );
}

const isMain =
  process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  main().catch((error: unknown) => {
    console.error(error);
    process.exit(1);
  });
}
