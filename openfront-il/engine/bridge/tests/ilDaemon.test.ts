/**
 * Tests for the live IL daemon. Like integration.test.ts these boot the real
 * vendored engine (not mocked): the in-process tests drive handleRequest
 * directly, and one subprocess test proves the stdio framing — that engine
 * logging never corrupts the one-JSON-line-per-reply stdout stream.
 */
import { spawn } from "node:child_process";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { AGENT_CLIENT_ID, handleRequest, IlSession } from "../src/ilDaemon";
import type { Reply, ResetReply, StepReply } from "../src/ilDaemon";

const projectRoot = path.resolve(import.meta.dirname, "..", "..", "..");

function expectOk<T extends Reply>(reply: Reply): T {
  if ("error" in reply) throw new Error(`daemon error: ${reply.error}`);
  return reply as T;
}

/** A land tile with no owner whose neighbors are all land — a safe spawn. */
function findSpawnTile(session: IlSession): number {
  const game = session.currentGame;
  if (game === null) throw new Error("no game");
  for (let ref = Math.floor((game.width() * game.height()) / 2); ref < game.width() * game.height(); ref++) {
    if (game.isLand(ref) && !game.hasOwner(ref) && game.neighbors(ref).every((n) => game.isLand(n))) {
      return ref;
    }
  }
  throw new Error("no spawn tile found");
}

async function stepUntilSpawnPhaseEnds(session: IlSession): Promise<StepReply> {
  for (let i = 0; i < 100; i++) {
    const reply = expectOk<StepReply>(await handleRequest(session, { op: "step" }));
    if (!reply.tick.spawnPhase) return reply;
  }
  throw new Error("spawn phase never ended");
}

function agentTiles(reply: StepReply): number {
  return reply.tick.players.find((p) => p.smallID === reply.me)?.numTilesOwned ?? 0;
}

describe("ilDaemon (in-process)", () => {
  it("resets into the spawn phase with a header, and seeds are deterministic", async () => {
    const reset = async (seed: string): Promise<ResetReply> =>
      expectOk<ResetReply>(await handleRequest(new IlSession(), { op: "reset", seed, bots: 4 }));

    const first = await reset("a");
    expect(first.header.kind).toBe("header");
    expect(first.header.map).toBe("Onion");
    expect(first.header.snapshotEvery).toBe(10);
    expect(first.tick.kind).toBe("tick");
    expect(first.tick.spawnPhase).toBe(true);
    expect(typeof first.me).toBe("number");

    expect((await reset("a")).header.gameId).toBe(first.header.gameId);
    expect((await reset("b")).header.gameId).not.toBe(first.header.gameId);
  }, 30_000);

  it("applies a spawn intent: the agent is alive, owns land and has a legality mask", async () => {
    const session = new IlSession();
    expectOk(await handleRequest(session, { op: "reset", seed: "spawn", bots: 4 }));

    const tile = findSpawnTile(session);
    const spawned = expectOk<StepReply>(
      await handleRequest(session, { op: "step", intents: [{ type: "spawn", tile }] }),
    );
    expect(spawned.rejected).toEqual([]);
    expect(spawned.tick.actions[AGENT_CLIENT_ID]).toEqual([
      { type: "spawn", tile, clientID: AGENT_CLIENT_ID },
    ]);

    const after = await stepUntilSpawnPhaseEnds(session);
    expect(after.alive).toBe(true);
    expect(agentTiles(after)).toBeGreaterThan(0);
    expect(after.tick.legal[AGENT_CLIENT_ID]).toBeDefined();
  }, 30_000);

  it("rejects a malformed intent without dropping the valid ones beside it", async () => {
    const session = new IlSession();
    expectOk(await handleRequest(session, { op: "reset", seed: "reject", bots: 4 }));

    const tile = findSpawnTile(session);
    const reply = expectOk<StepReply>(
      await handleRequest(session, {
        op: "step",
        intents: [{ type: "spawn", tile: "oops" }, { type: "spawn", tile }],
      }),
    );
    expect(reply.rejected).toHaveLength(1);
    expect(reply.rejected[0]?.index).toBe(0);
    expect(reply.rejected[0]?.reason).toContain("tile");
    expect(reply.tick.actions[AGENT_CLIENT_ID]).toHaveLength(1);
    expect(reply.alive).toBe(true);
  }, 30_000);

  it("leaves an agent that never spawns eliminated once the spawn phase ends", async () => {
    // Engine behavior, recorded so an engine bump that changes it is noticed:
    // in a Public game, not spawning means sitting the game out at 0 tiles.
    const session = new IlSession();
    expectOk(await handleRequest(session, { op: "reset", seed: "nospawn", bots: 4 }));

    const after = await stepUntilSpawnPhaseEnds(session);
    expect(after.alive).toBe(false);
    expect(agentTiles(after)).toBe(0);
    expect(after.tick.legal[AGENT_CLIENT_ID]).toBeUndefined();
  }, 30_000);

  it("replies {error} instead of throwing", async () => {
    expect(await handleRequest(new IlSession(), { op: "step" })).toHaveProperty("error");
    expect(await handleRequest(new IlSession(), { op: "reset", map: "Nowhere" })).toHaveProperty(
      "error",
    );

    const session = new IlSession();
    expectOk(await handleRequest(session, { op: "reset", seed: "errors", bots: 4 }));
    expect(await handleRequest(session, { op: "step", ticks: 0 })).toHaveProperty("error");
  }, 30_000);
});

describe("ilDaemon (subprocess)", () => {
  it("answers every request with exactly one JSON line and exits cleanly on shutdown", async () => {
    const child = spawn(
      path.join(projectRoot, "node_modules", ".bin", "tsx"),
      ["--tsconfig", "tsconfig.run.json", "engine/bridge/src/ilDaemon.ts"],
      { cwd: projectRoot, stdio: ["pipe", "pipe", "pipe"] },
    );
    let stdout = "";
    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });
    const exitCode = new Promise<number | null>((resolve) => child.on("exit", resolve));

    child.stdin.write(
      [
        JSON.stringify({ op: "reset", seed: "stdio", bots: 4 }),
        JSON.stringify({ op: "step" }),
        "not json",
        JSON.stringify({ op: "shutdown" }),
        "",
      ].join("\n"),
    );

    expect(await exitCode).toBe(0);
    const replies = stdout
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as Record<string, unknown>);
    expect(replies).toHaveLength(4);
    expect(replies[0]).toHaveProperty("header");
    expect(replies[1]).toHaveProperty("tick");
    expect(String(replies[2]?.error)).toContain("bad json");
    expect(replies[3]).toEqual({ ok: true });
  }, 60_000);
});
