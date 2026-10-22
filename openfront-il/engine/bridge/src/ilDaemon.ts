/**
 * Live engine daemon for the imitation-learning policy: boots a fresh game,
 * steps it with intents chosen by Python, and returns observations in the
 * exact SnapshotHeader/TickSnapshot format the policy was trained on.
 *
 * Observations are built by the same buildTerrainHeader/buildTickSnapshot
 * calls replayGame.ts uses, and games boot through the same
 * bootGameFromStartInfo the replay tests hash-verify — so a live
 * observation and a training snapshot are produced by identical code.
 *
 * One game per process. Protocol: one JSON request per stdin line, exactly
 * one JSON reply per stdout line, strictly alternating.
 *
 *   {"op":"reset", "map":"Onion", "seed":"x", "bots":8, ...}
 *     -> {"header": SnapshotHeader, "tick": TickSnapshot, "me": number}
 *   {"op":"step", "intents":[Intent...], "ticks":10}
 *     -> {"tick": TickSnapshot, "me": number, "alive": bool, "rejected": [...]}
 *   {"op":"shutdown"}
 *     -> {"ok": true}            (then the process exits)
 *
 * Any failure replies {"error": "..."} instead of exiting. Engine logging
 * is redirected to stderr so it can never corrupt the stdout stream.
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
} from "../../openfront-ai/openfront/src/core/game/Game";
import { GameUpdateType } from "../../openfront-ai/openfront/src/core/game/GameUpdates";
import type { Executor } from "../../openfront-ai/openfront/src/core/execution/ExecutionManager";
import { GameStartInfoSchema, IntentSchema } from "../../openfront-ai/openfront/src/core/Schemas";
import type { GameStartInfo } from "../../openfront-ai/openfront/src/core/Schemas";
import { simpleHash } from "../../openfront-ai/openfront/src/core/Util";
import { bootGameFromStartInfo, normalizeWinner } from "./replayGame";
import { buildTerrainHeader, buildTickSnapshot } from "./snapshotWriter";
import { ENGINE_ROOT } from "./types";
import type { SnapshotHeader, StampedIntent, TickSnapshot } from "./types";

/** The agent's clientID. Must match the engine's 8-alphanumeric ID format;
 * distinct from openfront-ai's AGENTRL1 so records from the two projects
 * can't be confused. */
export const AGENT_CLIENT_ID = "ILAGENT1";
const AGENT_USERNAME = "ILAgent";

/** Same default replayGame.ts uses, so live decisions are spaced like the
 * training snapshots unless deliberately overridden. */
const DEFAULT_SNAPSHOT_EVERY = Number(process.env.OPENFRONT_SNAPSHOT_EVERY ?? 10);

// ---------------------------------------------------------------------------
// Protocol types
// ---------------------------------------------------------------------------

export interface ResetOptions {
  map?: string;
  seed?: string;
  bots?: number;
  difficulty?: string;
  nations?: number | "default" | "disabled";
  gameType?: string;
  snapshotEvery?: number;
}

export type Request =
  | ({ op: "reset" } & ResetOptions)
  | { op: "step"; intents?: unknown[]; ticks?: number }
  | { op: "shutdown" };

export interface RejectedIntent {
  index: number;
  intent: unknown;
  reason: string;
}

export interface ResetReply {
  header: SnapshotHeader;
  tick: TickSnapshot;
  me: number;
}

export interface StepReply {
  tick: TickSnapshot;
  me: number;
  alive: boolean;
  rejected: RejectedIntent[];
}

export type Reply = ResetReply | StepReply | { ok: true } | { error: string };

// ---------------------------------------------------------------------------
// Game setup
// ---------------------------------------------------------------------------

/** Deterministic seed -> 8-char gameID. The engine seeds every PRNG from the
 * gameID, so the same seed reproduces the same nations, bots and spawns. */
export function seedToGameID(seed: string): string {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let h = simpleHash(`il-${seed}`);
  let out = "";
  for (let i = 0; i < 8; i++) {
    h = (h * 1103515245 + 12345) & 0x7fffffff;
    out += alphabet[h % alphabet.length];
  }
  return out;
}

/** The vendored engine's pinned upstream commit, recorded in the header's
 * gitCommit field the way replay headers record the archived game's. */
function engineCommit(): string {
  try {
    return fs.readFileSync(path.join(ENGINE_ROOT, "..", "openfront.commit"), "utf8").trim();
  } catch {
    return "unknown";
  }
}

/** Builds and schema-validates the GameStartInfo a live server would have
 * sent, with the agent as the only human. Defaults to a Public FFA game to
 * match the training data (configs/data.yaml) — Public also runs the fixed
 * spawn timer, where Singleplayer would freeze until the agent spawns. */
export function buildStartInfo(options: ResetOptions): GameStartInfo {
  return GameStartInfoSchema.parse({
    gameID: seedToGameID(options.seed ?? "0"),
    // Fixed rather than Date.now() so a seed fully determines the game.
    lobbyCreatedAt: 0,
    config: {
      gameMap: (options.map ?? "Onion") as GameMapType,
      gameMapSize: GameMapSize.Normal,
      gameMode: GameMode.FFA,
      gameType: (options.gameType ?? GameType.Public) as GameType,
      difficulty: (options.difficulty ?? Difficulty.Medium) as Difficulty,
      nations: options.nations ?? "default",
      bots: options.bots ?? 8,
      donateGold: true,
      donateTroops: true,
      infiniteGold: false,
      infiniteTroops: false,
      instantBuild: false,
      randomSpawn: false,
    },
    players: [{ clientID: AGENT_CLIENT_ID, username: AGENT_USERNAME, clanTag: null }],
  });
}

// ---------------------------------------------------------------------------
// Session: one live game
// ---------------------------------------------------------------------------

export class IlSession {
  private game: Game | null = null;
  private executor: Executor | null = null;
  private snapshotEvery = DEFAULT_SNAPSHOT_EVERY;
  /** Sticky once announced, matching replayGame.ts: every snapshot after a
   * win still carries it. */
  private winner: { kind: string; id: string } | null = null;

  /** The live game, for tests and diagnostics. */
  get currentGame(): Game | null {
    return this.game;
  }

  async reset(options: ResetOptions): Promise<ResetReply> {
    const startInfo = buildStartInfo(options);
    const { game, executor } = await bootGameFromStartInfo(startInfo);
    this.game = game;
    this.executor = executor;
    this.snapshotEvery = options.snapshotEvery ?? DEFAULT_SNAPSHOT_EVERY;
    this.winner = null;

    // Nations and bots are only added on the first tick; without it the
    // first observation would show an empty world.
    game.executeNextTick();

    const header = buildTerrainHeader(
      game,
      startInfo.gameID,
      engineCommit(),
      String(startInfo.config.gameMap),
      this.snapshotEvery,
    );
    return { header, tick: this.snapshot({}), me: this.agent(game).smallID() };
  }

  /** Applies the agent's intents as one turn, then advances `ticks` ticks
   * (stopping early on a win). Episode termination is left to the caller:
   * `alive` is false both before the agent spawns and after it dies, and
   * only the caller can tell those apart (via tick.spawnPhase). */
  step(rawIntents: unknown[], ticks: number = this.snapshotEvery): StepReply {
    const game = this.game;
    const executor = this.executor;
    if (game === null || executor === null) {
      throw new Error("step before reset");
    }
    if (!Number.isInteger(ticks) || ticks < 1) {
      throw new Error(`ticks must be a positive integer, got ${String(ticks)}`);
    }

    const stamped: StampedIntent[] = [];
    const rejected: RejectedIntent[] = [];
    rawIntents.forEach((intent, index) => {
      const parsed = IntentSchema.safeParse(intent);
      if (parsed.success) {
        stamped.push({ ...parsed.data, clientID: AGENT_CLIENT_ID } as StampedIntent);
      } else {
        const reason = parsed.error.issues
          .map((issue) => `${issue.path.join(".") || "(root)"}: ${issue.message}`)
          .join("; ");
        rejected.push({ index, intent, reason });
      }
    });

    if (stamped.length > 0) {
      game.addExecution(...executor.createExecs({ turnNumber: game.ticks(), intents: stamped }));
    }

    for (let t = 0; t < ticks; t++) {
      const updates = game.executeNextTick();
      const winUpdates = updates[GameUpdateType.Win] as { winner: unknown }[] | undefined;
      const firstWinUpdate = winUpdates?.[0];
      if (firstWinUpdate !== undefined) {
        this.winner = normalizeWinner(firstWinUpdate.winner);
        break;
      }
    }

    const agent = this.agent(game);
    return {
      tick: this.snapshot({ [AGENT_CLIENT_ID]: stamped }),
      me: agent.smallID(),
      alive: agent.isAlive(),
      rejected,
    };
  }

  private snapshot(actions: Record<string, StampedIntent[]>): TickSnapshot {
    if (this.game === null) throw new Error("no game");
    return buildTickSnapshot(this.game, new Set([AGENT_CLIENT_ID]), actions, this.winner);
  }

  private agent(game: Game): NonNullable<ReturnType<Game["playerByClientID"]>> {
    const player = game.playerByClientID(AGENT_CLIENT_ID);
    if (player === null) throw new Error(`agent ${AGENT_CLIENT_ID} missing from game`);
    return player;
  }
}

// ---------------------------------------------------------------------------
// Request handling + stdio loop
// ---------------------------------------------------------------------------

/** Exactly one reply per request; failures become {error} replies rather
 * than exceptions, so the caller always gets a line back. `shutdown` is
 * handled by the stdio loop, not here. */
export async function handleRequest(session: IlSession, request: Request): Promise<Reply> {
  try {
    switch (request.op) {
      case "reset":
        return await session.reset(request);
      case "step":
        return session.step(request.intents ?? [], request.ticks);
      default:
        return { error: `unknown op ${String((request as { op: unknown }).op)}` };
    }
  } catch (error: unknown) {
    return { error: error instanceof Error ? (error.stack ?? error.message) : String(error) };
  }
}

async function main(): Promise<void> {
  // Engine code logs through console.*; stdout is reserved for replies.
  const toStderr = (...args: unknown[]): void => {
    process.stderr.write(args.map(String).join(" ") + "\n");
  };
  console.log = console.info = console.warn = console.debug = toStderr;

  const write = (reply: Reply): void => {
    process.stdout.write(JSON.stringify(reply) + "\n");
  };

  const session = new IlSession();
  // Ends on `shutdown` or when stdin closes (the Python driver exited), so
  // the daemon never outlives its caller.
  for await (const line of readline.createInterface({ input: process.stdin })) {
    if (!line.trim()) continue;
    let request: Request;
    try {
      request = JSON.parse(line) as Request;
    } catch (error: unknown) {
      write({ error: `bad json: ${String(error)}` });
      continue;
    }
    if (request.op === "shutdown") {
      write({ ok: true });
      break;
    }
    write(await handleRequest(session, request));
  }
  process.exit(0);
}

const isMain =
  process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  main().catch((error: unknown) => {
    console.error(error);
    process.exit(1);
  });
}
