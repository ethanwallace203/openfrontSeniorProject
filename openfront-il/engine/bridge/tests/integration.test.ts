/**
 * End-to-end replay tests against two real, live-fetched games (checked
 * into fixtures/). These boot the actual vendored engine — not mocked —
 * so they're slower than a unit test, but this is the only test that
 * actually proves the boot sequence, tick loop, and hash verification
 * work against real data.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { replayGame } from "../src/replayGame";
import { snapshotPath } from "../src/types";
import type { GameRecord } from "../src/types";

const fixturesDir = path.join(import.meta.dirname, "fixtures");

function loadFixture(gameId: string): GameRecord {
  const raw = fs.readFileSync(path.join(fixturesDir, `${gameId}.json`), "utf8");
  return JSON.parse(raw) as GameRecord;
}

describe("replayGame", () => {
  it("achieves full hash parity on a Singleplayer/bots game", async () => {
    const record = loadFixture("USR3SJiv");

    const result = await replayGame(record, 10);

    expect(result.ok).toBe(true);
    expect(result.hashesChecked).toBeGreaterThan(0);
    expect(result.snapshotsWritten).toBeGreaterThan(0);

    const lines = fs
      .readFileSync(snapshotPath("USR3SJiv"), "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as Record<string, unknown>);
    expect(lines[0]?.kind).toBe("header");
    expect(lines.length).toBe(result.snapshotsWritten + 1);
    expect(lines.every((line, i) => (i === 0 ? true : line.kind === "tick"))).toBe(true);
  }, 30_000);

  it("achieves full hash parity on a real Public 1v1 human game with purchased tribe names", async () => {
    // This fixture is the regression test for a real bug: bootGame() used
    // to omit info.tribes (purchased bot tribe names) when constructing
    // Executor, which silently skipped a PRNG-consuming shuffleArray call
    // inside TribeSpawner.spawnTribes() whenever a game actually had
    // purchased names — desyncing every tribe's spawn tile and ID from
    // that point on. See replayGame.ts's bootGame() comment on
    // purchasedTribeNames for the full mechanism.
    const record = loadFixture("vmVhD8B3");

    const result = await replayGame(record, 10);

    expect(result.ok).toBe(true);
    expect(result.hashesChecked).toBeGreaterThan(0);
    expect(result.snapshotsWritten).toBeGreaterThan(0);
  }, 30_000);
});
