import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { fromCliArgs, fromSeedFile, loadDataConfig } from "../src/gameIdSource";

describe("fromCliArgs", () => {
  it("drops arguments starting with '-' but keeps everything else, including flag values", () => {
    // fromCliArgs only recognizes the "-" prefix itself — it doesn't know
    // which flags take a following value. resolveGameIds() is responsible
    // for stripping a flag's value before calling this (see its own
    // "--config" filtering), and never calls this at all on the
    // "--seed-file" branch.
    expect(fromCliArgs(["abc123", "--seed-file", "foo.txt", "def456"])).toEqual([
      "abc123",
      "foo.txt",
      "def456",
    ]);
  });

  it("drops a bare flag with no value", () => {
    expect(fromCliArgs(["--config"])).toEqual([]);
  });
});

describe("fromSeedFile", () => {
  const tempFiles: string[] = [];

  afterEach(() => {
    for (const file of tempFiles.splice(0)) fs.rmSync(file, { force: true });
  });

  function writeSeedFile(contents: string): string {
    const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "seed-")), "seed.txt");
    fs.writeFileSync(file, contents);
    tempFiles.push(file);
    return file;
  }

  it("reads one gameId per line, ignoring blanks and comments", () => {
    const file = writeSeedFile("abc123\n\n# a comment\ndef456\n  \nghi789");
    expect(fromSeedFile(file)).toEqual(["abc123", "def456", "ghi789"]);
  });
});

describe("loadDataConfig", () => {
  it("loads and validates the project's own configs/data.yaml", () => {
    const configPath = path.join(import.meta.dirname, "..", "..", "..", "configs", "data.yaml");
    const config = loadDataConfig(configPath);
    expect(config.gameType).toBe("Public");
    expect(config.targetGameCount).toBeGreaterThan(0);
  });

  it("rejects a config missing required fields", () => {
    const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "cfg-")), "data.yaml");
    fs.writeFileSync(file, "gameType: Public\n");
    expect(() => loadDataConfig(file)).toThrow();
  });
});
