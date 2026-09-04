import * as fs from "node:fs";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { GameRecordSchema, rawGamePath, snapshotPath } from "../src/types";

const fixturesDir = path.join(import.meta.dirname, "fixtures");

describe("GameRecordSchema", () => {
  it("validates a real fetched game record", () => {
    const raw = fs.readFileSync(path.join(fixturesDir, "USR3SJiv.json"), "utf8");
    const result = GameRecordSchema.safeParse(JSON.parse(raw));
    expect(result.success).toBe(true);
  });

  it("rejects a record missing required fields", () => {
    const result = GameRecordSchema.safeParse({ info: {} });
    expect(result.success).toBe(false);
  });
});

describe("path constants", () => {
  it("builds paths under the configured data directory", () => {
    // setup.ts points OPENFRONT_DATA_DIR at an isolated temp dir for tests,
    // so only the raw/snapshots subpath is asserted here, not "data/".
    expect(rawGamePath("abc123")).toMatch(/[/\\]raw[/\\]abc123\.json$/);
    expect(snapshotPath("abc123")).toMatch(/[/\\]snapshots[/\\]abc123\.jsonl$/);
  });
});
