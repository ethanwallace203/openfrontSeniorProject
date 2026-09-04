import * as fs from "node:fs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fetchGame } from "../src/fetchGame";
import type { GameRecord } from "../src/types";
import { RAW_DIR, rawGamePath } from "../src/types";
import fixtureRecord from "./fixtures/USR3SJiv.json";

describe("fetchGame", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    fs.rmSync(RAW_DIR, { recursive: true, force: true });
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("writes a valid response to data/raw/{gameId}.json", async () => {
    global.fetch = vi.fn(
      () => new Response(JSON.stringify(fixtureRecord), { status: 200 }),
    ) as unknown as typeof fetch;

    const result = await fetchGame("USR3SJiv");

    expect(result).toBe("written");
    expect(fs.existsSync(rawGamePath("USR3SJiv"))).toBe(true);
    const written = JSON.parse(fs.readFileSync(rawGamePath("USR3SJiv"), "utf8")) as GameRecord;
    expect(written.info.gameID).toBe("USR3SJiv");
  });

  it("never overwrites an existing raw file", async () => {
    fs.mkdirSync(RAW_DIR, { recursive: true });
    fs.writeFileSync(rawGamePath("USR3SJiv"), '{"already":"here"}');
    global.fetch = vi.fn() as unknown as typeof fetch;

    const result = await fetchGame("USR3SJiv");

    expect(result).toBe("skipped-exists");
    expect(global.fetch).not.toHaveBeenCalled();
    expect(fs.readFileSync(rawGamePath("USR3SJiv"), "utf8")).toBe('{"already":"here"}');
  });

  it("fails without writing when the response doesn't match GameRecordSchema", async () => {
    global.fetch = vi.fn(
      () => new Response(JSON.stringify({ not: "a game record" }), { status: 200 }),
    ) as unknown as typeof fetch;

    const result = await fetchGame("bad0000");

    expect(result).toBe("failed");
    expect(fs.existsSync(rawGamePath("bad0000"))).toBe(false);
  });

  it("fails without writing when every retry returns a non-ok status", async () => {
    // fetchGame.ts reads its retry/backoff env vars once at module import
    // time, so this exercises the real default retry count — fake timers
    // skip the real backoff delay between attempts instead of waiting.
    vi.useFakeTimers();
    global.fetch = vi.fn(() => new Response("", { status: 500 })) as unknown as typeof fetch;

    const resultPromise = fetchGame("down0000");
    await vi.runAllTimersAsync();
    const result = await resultPromise;

    expect(result).toBe("failed");
    expect(fs.existsSync(rawGamePath("down0000"))).toBe(false);
    vi.useRealTimers();
  });
});
