/**
 * Live, headless singleplayer game for the agent: one agent-controlled
 * human vs the map's nations and N tribes (bots), driven step-by-step
 * over stdio by the Python agent (src/openfront_il/agent/).
 *
 * Unlike replayGame.ts (which replays a finished record from disk), this
 * runs a brand-new game whose human seat is played by the policy. It
 * emits the exact same SnapshotHeader/TickSnapshot JSON that replayGame
 * writes to data/snapshots/*.jsonl — built by the same snapshotWriter.ts
 * functions — so the Python side parses live observations with the same
 * pydantic models (and the policy sees the same feature encoding) it was
 * trained on.
 *
 * Game setup mirrors engine/openfront-ai/bridge/session.ts (EnvSession),
 * used as a reference only — that file lives in the read-only submodule.
 *
 * Protocol: one JSON object per line on stdin; one JSON object per line on
 * stdout. Every stdout line has a "kind" field.
 *
 *   -> {"op":"reset","map":"World","seed":"s1","bots":8,"difficulty":"Medium",
 *       "nations":"default","snapshotEvery":10}
 *   <- {"kind":"header",...}            SnapshotHeader
 *   <- {"kind":"tick",...}              TickSnapshot (first post-spawn tick)
 *
 *   -> {"op":"step","intents":[Intent...],"ticks":10}
 *   <- {"kind":"tick",...}              TickSnapshot after the ticks ran
 *                                       (stops early on a win)
 *
 *   -> {"op":"save","path":"data/agent_games/x.json"}
 *   <- {"kind":"saved","path":...,"gameID":...,"turns":N}
 *
 *   -> {"op":"close"}                   exits
 *
 *   <- {"kind":"error","message":...}   on any failed op; the process stays up
 *
 * The agent spawns randomly (randomSpawn: true) — the policy was trained
 * only on post-spawn-phase snapshots (replayGame.ts never snapshots the
 * spawn phase), so it has no learned spawn behavior. reset() runs ticks
 * until the spawn phase ends before returning the first observation.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import * as readline from "node:readline";
import { pathToFileURL } from "node:url";
import {
  Difficulty,
  Game,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  PlayerInfo,
  PlayerType,
  UnitType,
} from "../../openfront-ai/openfront/src/core/game/Game";
import { createGame } from "../../openfront-ai/openfront/src/core/game/GameImpl";
import { GameUpdateType } from "../../openfront-ai/openfront/src/core/game/GameUpdates";
import { createNationsForGame } from "../../openfront-ai/openfront/src/core/game/NationCreation";
import { Config } from "../../openfront-ai/openfront/src/core/configuration/Config";
import { Executor } from "../../openfront-ai/openfront/src/core/execution/ExecutionManager";
import { DoomsdayClockExecution } from "../../openfront-ai/openfront/src/core/execution/DoomsdayClockExecution";
import { RecomputeRailClusterExecution } from "../../openfront-ai/openfront/src/core/execution/RecomputeRailClusterExecution";
import { WinCheckExecution } from "../../openfront-ai/openfront/src/core/execution/WinCheckExecution";
import { PseudoRandom } from "../../openfront-ai/openfront/src/core/PseudoRandom";
import { simpleHash } from "../../openfront-ai/openfront/src/core/Util";
import type {
  GameConfig,
  GameStartInfo,
  Intent,
} from "../../openfront-ai/openfront/src/core/Schemas";
import { loadFreshTerrain, normalizeWinner } from "./replayGame";
import { buildTerrainHeader, buildTickSnapshot } from "./snapshotWriter";
import type { StampedIntent, TickSnapshot } from "./types";

export const AGENT_CLIENT_ID = "AGENTIL1";
const AGENT_USERNAME = "Agent";
const DEFAULT_SNAPSHOT_EVERY = 10;
// Safety cap so a misconfigured game can't spin forever in reset().
const MAX_SPAWN_PHASE_TICKS = 10_000;

// stdout is the protocol channel — route every engine log line to stderr
// so it can never corrupt the JSON stream.
console.log = console.info = console.warn = console.debug = (...args: unknown[]) => {
  process.stderr.write(args.map(String).join(" ") + "\n");
};

/** Accepts either the enum key ("WorldMap"-style, as session.ts does) or
 * the enum value (the display string, as game records store it). */
function resolveMap(name: string): GameMapType {
  const byKey = GameMapType[name as keyof typeof GameMapType];
  if (byKey !== undefined) return byKey;
  if ((Object.values(GameMapType) as string[]).includes(name)) return name as GameMapType;
  throw new Error(`unknown map: ${name}`);
}

function resolveDifficulty(name: string): Difficulty {
  const value = Difficulty[name as keyof typeof Difficulty];
  if (value === undefined) throw new Error(`unknown difficulty: ${name}`);
  return value;
}

/** Same 8-char alphanumeric shape real game IDs have, derived from the seed
 * so a given seed always produces the same game. */
function seedToGameID(seed: string): string {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let h = simpleHash(`il-${seed}`);
  let out = "";
  for (let i = 0; i < 8; i++) {
    h = (h * 1103515245 + 12345) & 0x7fffffff;
    out += alphabet[h % alphabet.length];
  }
  return out;
}

export interface ResetOptions {
  map: string;
  seed: string;
  bots: number;
  difficulty: string;
  nations?: number | "default" | "disabled";
  snapshotEvery?: number;
}

export class LiveGame {
  private game!: Game;
  private executor!: Executor;
  private gameID = "";
  private gameConfig!: GameConfig;
  private startTime = 0;
  private turns: { turnNumber: number; intents: StampedIntent[] }[] = [];
  private winner: { kind: string; id: string } | null = null;
  private readonly agentIDs: ReadonlySet<string> = new Set([AGENT_CLIENT_ID]);
  snapshotEvery = DEFAULT_SNAPSHOT_EVERY;

  async reset(options: ResetOptions): Promise<[object, TickSnapshot]> {
    const mapType = resolveMap(options.map);
    this.snapshotEvery = options.snapshotEvery ?? DEFAULT_SNAPSHOT_EVERY;

    const gameConfig: GameConfig = {
      gameMap: mapType,
      gameMapSize: GameMapSize.Normal,
      gameMode: GameMode.FFA,
      gameType: GameType.Singleplayer,
      difficulty: resolveDifficulty(options.difficulty),
      nations: options.nations ?? "default",
      donateGold: true,
      donateTroops: true,
      bots: options.bots,
      infiniteGold: false,
      infiniteTroops: false,
      instantBuild: false,
      randomSpawn: true,
    };
    const gameID = seedToGameID(options.seed);
    const gameStart: GameStartInfo = {
      gameID,
      lobbyCreatedAt: Date.now(),
      config: gameConfig,
      players: [{ clientID: AGENT_CLIENT_ID, username: AGENT_USERNAME, clanTag: null }],
    };

    this.gameID = gameID;
    this.gameConfig = gameConfig;
    this.startTime = Date.now();
    this.turns = [];
    this.winner = null;

    const config = new Config(gameConfig, null, false);
    const terrain = await loadFreshTerrain(mapType, gameConfig.gameMapSize);
    const random = new PseudoRandom(simpleHash(gameID));

    // Humans consume random.nextID() before nations — same order as
    // replayGame.ts's bootGame() and the live engine.
    const humans = gameStart.players.map(
      (p) => new PlayerInfo(p.username, PlayerType.Human, p.clientID, random.nextID()),
    );
    const nations = createNationsForGame(
      gameStart,
      terrain.nations,
      terrain.additionalNations,
      humans.length,
      random,
    );

    this.game = createGame(
      humans,
      nations,
      terrain.gameMap,
      terrain.miniGameMap,
      config,
      terrain.teamGameSpawnAreas,
    );
    this.executor = new Executor(this.game, gameID, AGENT_CLIENT_ID);

    // Singleplayer: no SpawnTimerExecution — the spawn phase ends when the
    // human spawns (SpawnExecution), which randomSpawn triggers for us.
    if (config.spawnNations()) {
      this.game.addExecution(...this.executor.nationExecutions());
    }
    if (config.isRandomSpawn()) {
      this.game.addExecution(...this.executor.spawnPlayers());
    }
    if (config.bots() > 0) {
      this.game.addExecution(...this.executor.spawnTribes(config.bots()));
    }
    this.game.addExecution(new WinCheckExecution());
    if (config.doomsdayClockConfig().enabled) {
      this.game.addExecution(new DoomsdayClockExecution());
    }
    if (!config.isUnitDisabled(UnitType.Factory)) {
      this.game.addExecution(new RecomputeRailClusterExecution(this.game.railNetwork()));
    }

    this.game.executeNextTick();
    let spawnTicks = 0;
    while (this.game.inSpawnPhase()) {
      if (++spawnTicks > MAX_SPAWN_PHASE_TICKS) {
        throw new Error(`spawn phase did not end within ${MAX_SPAWN_PHASE_TICKS} ticks`);
      }
      this.runTick();
    }

    const header = buildTerrainHeader(
      this.game,
      gameID,
      "live",
      String(mapType),
      this.snapshotEvery,
    );
    return [header, this.snapshot({})];
  }

  step(intents: Intent[], ticks: number): TickSnapshot {
    if (this.game === undefined) throw new Error("step before reset");

    const stamped = intents.map((intent) => ({
      ...intent,
      clientID: AGENT_CLIENT_ID,
    })) as StampedIntent[];
    if (stamped.length > 0) {
      const turn = { turnNumber: this.game.ticks(), intents: stamped };
      this.turns.push(turn);
      this.game.addExecution(...this.executor.createExecs(turn));
    }

    for (let t = 0; t < ticks && this.winner === null; t++) {
      this.runTick();
    }
    return this.snapshot(stamped.length > 0 ? { [AGENT_CLIENT_ID]: stamped } : {});
  }

  /** Writes this game in the archive GameRecord shape (same as
   * session.ts's saveRecord), so it can be inspected or replayed later. */
  save(outPath: string): object {
    const end = Date.now();
    const record = {
      info: {
        gameID: this.gameID,
        lobbyCreatedAt: this.startTime,
        config: this.gameConfig,
        players: [
          {
            clientID: AGENT_CLIENT_ID,
            username: AGENT_USERNAME,
            clanTag: null,
            persistentID: null,
            stats: {},
          },
        ],
        start: this.startTime,
        end,
        duration: Math.floor((end - this.startTime) / 1000),
        num_turns: this.game.ticks(),
        winner: this.winner === null ? undefined : [this.winner.kind, this.winner.id],
        lobbyFillTime: 0,
      },
      version: "v0.0.2",
      gitCommit: "DEV",
      subdomain: "il",
      domain: "localhost",
      turns: this.turns,
    };
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.writeFileSync(outPath, JSON.stringify(record));
    return { kind: "saved", path: outPath, gameID: this.gameID, turns: this.turns.length };
  }

  private runTick(): void {
    const updates = this.game.executeNextTick();
    const winUpdates = updates[GameUpdateType.Win] as { winner: unknown }[] | undefined;
    const first = winUpdates?.[0];
    if (first !== undefined && this.winner === null) {
      this.winner = normalizeWinner(first.winner);
    }
  }

  private snapshot(actions: Record<string, StampedIntent[]>): TickSnapshot {
    return buildTickSnapshot(this.game, this.agentIDs, actions, this.winner);
  }
}

interface Message {
  op?: string;
  map?: string;
  seed?: string;
  bots?: number;
  difficulty?: string;
  nations?: number | "default" | "disabled";
  snapshotEvery?: number;
  intents?: Intent[];
  ticks?: number;
  path?: string;
}

function write(obj: object): void {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

async function main(): Promise<void> {
  const live = new LiveGame();
  const rl = readline.createInterface({ input: process.stdin });

  for await (const line of rl) {
    if (!line.trim()) continue;
    try {
      const msg = JSON.parse(line) as Message;
      switch (msg.op) {
        case "reset": {
          const [header, tick] = await live.reset({
            map: msg.map ?? "World",
            seed: msg.seed ?? String(Date.now()),
            bots: msg.bots ?? 0,
            difficulty: msg.difficulty ?? "Medium",
            nations: msg.nations,
            snapshotEvery: msg.snapshotEvery,
          });
          write(header);
          write(tick);
          break;
        }
        case "step":
          write(live.step(msg.intents ?? [], msg.ticks ?? live.snapshotEvery));
          break;
        case "save":
          if (msg.path === undefined) throw new Error("save requires a path");
          write(live.save(msg.path));
          break;
        case "close":
          rl.close();
          return;
        default:
          throw new Error(`unknown op: ${String(msg.op)}`);
      }
    } catch (error: unknown) {
      write({
        kind: "error",
        message: error instanceof Error ? (error.stack ?? error.message) : String(error),
      });
    }
  }
}

const isMain =
  process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  main().catch((error: unknown) => {
    process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
    process.exit(1);
  });
}
