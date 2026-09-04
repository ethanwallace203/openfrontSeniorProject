/**
 * Builds the SnapshotHeader/TickSnapshot records defined in types.ts from a
 * live `Game` instance, and appends them as JSONL to
 * data/snapshots/{gameId}.jsonl. Called by replayGame.ts once per game
 * (header) and once per snapshot tick (TickSnapshot).
 */
import * as fs from "node:fs";
import { gzipSync } from "node:zlib";
import type {
  Attack,
  BuildableUnit,
  Game,
  MutableAlliance,
  Player,
  Unit,
} from "../../openfront-ai/openfront/src/core/game/Game";
import type {
  AllianceSnapshot,
  AttackSnapshot,
  LegalityMask,
  PlayerScalars,
  SnapshotHeader,
  StampedIntent,
  TickSnapshot,
  UnitSnapshot,
} from "./types";
import { SNAPSHOTS_DIR, snapshotPath } from "./types";

export class SnapshotWriter {
  private readonly fd: number;

  constructor(gameId: string) {
    fs.mkdirSync(SNAPSHOTS_DIR, { recursive: true });
    this.fd = fs.openSync(snapshotPath(gameId), "wx");
  }

  writeHeader(header: SnapshotHeader): void {
    fs.writeSync(this.fd, `${JSON.stringify(header)}\n`);
  }

  writeTick(tick: TickSnapshot): void {
    fs.writeSync(this.fd, `${JSON.stringify(tick)}\n`);
  }

  close(): void {
    fs.closeSync(this.fd);
  }
}

export function buildTerrainHeader(
  game: Game,
  gameId: string,
  gitCommit: string,
  map: string,
  snapshotEvery: number,
): SnapshotHeader {
  const width = game.width();
  const height = game.height();
  const terrain = new Uint8Array(width * height);
  for (let ref = 0; ref < terrain.length; ref++) {
    terrain[ref] = game.terrainByte(ref);
  }
  return {
    kind: "header",
    formatVersion: 1,
    gameId,
    gitCommit,
    map,
    width,
    height,
    terrainGzipBase64: gzipSync(
      Buffer.from(terrain.buffer, terrain.byteOffset, terrain.byteLength),
    ).toString("base64"),
    snapshotEvery,
  };
}

function playerScalars(player: Player): PlayerScalars {
  return {
    clientID: player.clientID(),
    smallID: player.smallID(),
    username: player.name(),
    type: player.type(),
    team: player.team(),
    isAlive: player.isAlive(),
    isTraitor: player.isTraitor(),
    isDisconnected: player.isDisconnected(),
    numTilesOwned: player.numTilesOwned(),
    gold: player.gold().toString(),
    troops: player.troops(),
  };
}

function unitSnapshot(game: Game, unit: Unit): UnitSnapshot {
  const tile = unit.tile();
  const targetTile = unit.targetTile();
  return {
    id: unit.id(),
    type: unit.type(),
    ownerSmallID: unit.owner().smallID(),
    x: game.x(tile),
    y: game.y(tile),
    targetX: targetTile === undefined ? null : game.x(targetTile),
    targetY: targetTile === undefined ? null : game.y(targetTile),
    health: unit.health(),
    maxHealth: unit.maxHealth(),
    troops: unit.troops(),
    level: unit.level(),
    isUnderConstruction: unit.isUnderConstruction(),
  };
}

function attackSnapshot(game: Game, attack: Attack): AttackSnapshot {
  const sourceTile = attack.sourceTile();
  return {
    id: attack.id(),
    attackerSmallID: attack.attacker().smallID(),
    // Both Player and TerraNullius expose smallID(); TerraNullius's is the
    // engine's reserved "unowned land" sentinel (0).
    targetSmallID: attack.target().smallID(),
    troops: attack.troops(),
    retreating: attack.retreating(),
    sourceX: sourceTile === null ? null : game.x(sourceTile),
    sourceY: sourceTile === null ? null : game.y(sourceTile),
  };
}

function allianceSnapshot(alliance: MutableAlliance): AllianceSnapshot {
  return {
    requestorSmallID: alliance.requestor().smallID(),
    recipientSmallID: alliance.recipient().smallID(),
    expiresAtTick: alliance.expiresAt(),
  };
}

/** Whether any of the player's border tiles has an unowned neighbor —
 * i.e. there is somewhere left to expand into by attacking terra nullius.
 * There is no single engine call for this (the real client only ever
 * checks a specific hovered tile); this is the cheapest honest
 * player-level approximation using the same border-tile set the client's
 * own border rendering uses. */
function canExpand(game: Game, player: Player): boolean {
  for (const tile of player.borderTiles()) {
    for (const neighbor of game.neighbors(tile)) {
      if (!game.hasOwner(neighbor)) return true;
    }
  }
  return false;
}

/** Aggregates the scattered `can*` predicate methods on Player into one
 * legality mask. Deliberately coarser than a full tile-level mask — see
 * LegalityMask's doc comment in types.ts for why. */
export function extractLegality(game: Game, player: Player): LegalityMask {
  const others = game.players().filter((p) => p.clientID() !== player.clientID());

  const buildableUnitTypesList: string[] = player
    .buildableUnits(null)
    .filter((buildable: BuildableUnit) => buildable.canBuild !== false)
    .map((buildable: BuildableUnit) => buildable.type);

  return {
    attackablePlayerSmallIDs: others
      .filter((other) => player.canAttackPlayer(other))
      .map((other) => other.smallID()),
    canExpand: canExpand(game, player),
    allianceRequestablePlayerSmallIDs: others
      .filter((other) => player.canSendAllianceRequest(other))
      .map((other) => other.smallID()),
    allianceAcceptablePlayerSmallIDs: player
      .incomingAllianceRequests()
      .map((request) => request.requestor().smallID()),
    breakableAlliancePlayerSmallIDs: others
      .filter((other) => player.isAlliedWith(other))
      .map((other) => other.smallID()),
    donatableGoldPlayerSmallIDs: others
      .filter((other) => player.canDonateGold(other))
      .map((other) => other.smallID()),
    donatableTroopsPlayerSmallIDs: others
      .filter((other) => player.canDonateTroops(other))
      .map((other) => other.smallID()),
    embargoablePlayerSmallIDs: others
      .filter((other) => !player.hasEmbargoAgainst(other))
      .map((other) => other.smallID()),
    targetablePlayerSmallIDs: others
      .filter((other) => player.canTarget(other))
      .map((other) => other.smallID()),
    buildableUnitTypes: buildableUnitTypesList,
    canDeleteUnit: player.canDeleteUnit(),
    canEmbargoAll: player.canEmbargoAll(),
  };
}

export function buildTickSnapshot(
  game: Game,
  humanClientIDs: ReadonlySet<string>,
  actionsByClient: Record<string, StampedIntent[]>,
  winner: { kind: string; id: string } | null,
): TickSnapshot {
  const players = game.allPlayers().map(playerScalars);
  const units: UnitSnapshot[] = [];
  const attacks: AttackSnapshot[] = [];
  const alliances: AllianceSnapshot[] = [];
  const legal: Record<string, LegalityMask> = {};

  for (const player of game.players()) {
    for (const unit of player.units()) {
      units.push(unitSnapshot(game, unit));
    }
    for (const attack of player.outgoingAttacks()) {
      attacks.push(attackSnapshot(game, attack));
    }
    for (const alliance of player.alliances()) {
      // Alliances are symmetric (requestor/recipient); only record once,
      // from the requestor's iteration, to avoid duplicates.
      if (alliance.requestor().clientID() === player.clientID()) {
        alliances.push(allianceSnapshot(alliance));
      }
    }
    const clientID = player.clientID();
    if (clientID !== null && humanClientIDs.has(clientID) && player.isAlive()) {
      legal[clientID] = extractLegality(game, player);
    }
  }

  return {
    kind: "tick",
    tick: game.ticks(),
    spawnPhase: game.inSpawnPhase(),
    tileStateGzipBase64: gzipSync(
      Buffer.from(
        game.tileStateBuffer().buffer,
        game.tileStateBuffer().byteOffset,
        game.tileStateBuffer().byteLength,
      ),
    ).toString("base64"),
    players,
    units,
    attacks,
    alliances,
    actions: actionsByClient,
    legal,
    winner,
  };
}
