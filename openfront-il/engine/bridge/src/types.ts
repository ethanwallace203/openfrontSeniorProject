/**
 * Shared types for the two artifacts this bridge produces:
 *
 * 1. The raw game record (data/raw/{gameId}.json) — not our format, we just
 *    re-export the real engine's own canonical Zod schema for it, so there
 *    is zero drift between what we validate against and what the engine
 *    itself considers a valid GameRecord.
 * 2. The snapshot (data/snapshots/{gameId}.jsonl) — this IS our own format,
 *    invented by this bridge. Its Python mirror lives at
 *    src/openfront_il/replay/snapshot_schema.py; see that file's module
 *    docstring for the hand-sync/drift-test contract between the two.
 */
import type { GameRecord, StampedIntent } from "../../openfront-ai/openfront/src/core/Schemas";
import { GameRecordSchema } from "../../openfront-ai/openfront/src/core/Schemas";

export { GameRecordSchema };
export type { GameRecord, StampedIntent };

// ---------------------------------------------------------------------------
// Path constants — the single source of truth for data/'s layout on the
// Node side. Mirrors src/openfront_il/utils/paths.py. Never hardcode a
// `data/...` path string anywhere else in engine/bridge.
// ---------------------------------------------------------------------------

import * as path from "node:path";

const PROJECT_ROOT = path.resolve(import.meta.dirname, "..", "..", "..");

export const DATA_DIR = process.env.OPENFRONT_DATA_DIR
  ? path.resolve(PROJECT_ROOT, process.env.OPENFRONT_DATA_DIR)
  : path.join(PROJECT_ROOT, "data");

export const RAW_DIR = path.join(DATA_DIR, "raw");
export const SNAPSHOTS_DIR = path.join(DATA_DIR, "snapshots");

export const ENGINE_ROOT = path.join(PROJECT_ROOT, "engine", "openfront-ai", "openfront");

export function rawGamePath(gameId: string): string {
  return path.join(RAW_DIR, `${gameId}.json`);
}

export function snapshotPath(gameId: string): string {
  return path.join(SNAPSHOTS_DIR, `${gameId}.jsonl`);
}

// ---------------------------------------------------------------------------
// Snapshot schema (our own format)
// ---------------------------------------------------------------------------

/**
 * The first line of every snapshot JSONL file. Written once per game.
 * Terrain never changes tick-to-tick, so it is not repeated on every
 * snapshot line — only the dynamic tile state (ownership/fallout/defense
 * bits) is.
 */
export interface SnapshotHeader {
  kind: "header";
  formatVersion: 1;
  gameId: string;
  gitCommit: string;
  map: string;
  width: number;
  height: number;
  /**
   * Gzip-then-base64 of the raw Uint8Array terrain bytes (one byte per
   * tile, row-major). Measured on a real 2904x1672 map: an uncompressed
   * per-tick tile-state buffer alone put a 470-tick singleplayer game's
   * snapshot file at 620MB — gzip first, base64 second, applied to every
   * tile buffer in this format (this field and TickSnapshot.tileState).
   */
  terrainGzipBase64: string;
  snapshotEvery: number;
}

export interface PlayerScalars {
  clientID: string | null;
  smallID: number;
  username: string;
  type: string;
  team: string | null;
  isAlive: boolean;
  isTraitor: boolean;
  isDisconnected: boolean;
  numTilesOwned: number;
  /** Gold is a bigint in the engine; serialized as a decimal string. */
  gold: string;
  troops: number;
}

export interface UnitSnapshot {
  id: number;
  type: string;
  ownerSmallID: number;
  x: number;
  y: number;
  targetX: number | null;
  targetY: number | null;
  health: number;
  maxHealth: number;
  troops: number;
  level: number;
  isUnderConstruction: boolean;
}

export interface AttackSnapshot {
  id: string;
  attackerSmallID: number;
  /** smallID of the defending player; 0 (TerraNullius) for unowned land. */
  targetSmallID: number;
  troops: number;
  retreating: boolean;
  sourceX: number | null;
  sourceY: number | null;
}

export interface AllianceSnapshot {
  requestorSmallID: number;
  recipientSmallID: number;
  expiresAtTick: number;
}

/**
 * A coarse, player/unit-type-level legality mask — NOT a full tile-level
 * mask (enumerating every legal (player, tile) pair every tick is
 * O(width*height*players) and not practical at this scale). Tile-level
 * legality for attack/build actions is left for
 * src/openfront_il/features/action_encoder.py to derive by joining this
 * mask against tile ownership, which it already has from the tile state.
 */
export interface LegalityMask {
  attackablePlayerSmallIDs: number[];
  canExpand: boolean;
  allianceRequestablePlayerSmallIDs: number[];
  allianceAcceptablePlayerSmallIDs: number[];
  breakableAlliancePlayerSmallIDs: number[];
  donatableGoldPlayerSmallIDs: number[];
  donatableTroopsPlayerSmallIDs: number[];
  embargoablePlayerSmallIDs: number[];
  targetablePlayerSmallIDs: number[];
  buildableUnitTypes: string[];
  canDeleteUnit: boolean;
  canEmbargoAll: boolean;
}

/**
 * One per-tick record. Written every `snapshotEvery` ticks (outside the
 * spawn phase). `actions` and `legal` are keyed by clientID and only
 * present for currently-alive human players — nations/bots have no
 * clientID and are never the imitation target.
 */
export interface TickSnapshot {
  kind: "tick";
  tick: number;
  spawnPhase: boolean;
  /** Gzip-then-base64 of the raw little-endian Uint16Array
   * (game.tileStateBuffer()) — see SnapshotHeader.terrainGzipBase64 for why. */
  tileStateGzipBase64: string;
  players: PlayerScalars[];
  units: UnitSnapshot[];
  attacks: AttackSnapshot[];
  alliances: AllianceSnapshot[];
  /** Intents each living human issued during [tick, tick+snapshotEvery). */
  actions: Record<string, StampedIntent[]>;
  /** Legality mask computed at this tick, before those actions were taken. */
  legal: Record<string, LegalityMask>;
  /** Normalized winner announcement, if the engine emitted one this tick. */
  winner: { kind: string; id: string } | null;
}

export type SnapshotLine = SnapshotHeader | TickSnapshot;
