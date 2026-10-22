"""Pydantic mirror of the snapshot format ``engine/bridge/src/types.ts``
defines and ``engine/bridge/src/snapshotWriter.ts`` writes to
``data/snapshots/{gameId}.jsonl``.

``types.ts`` is the source of truth — it's what actually writes the
files — and this is a hand-maintained mirror kept in sync manually. There
is no shared schema language between TypeScript and Python in this
prototype, so the two definitions can drift silently if one changes
without the other. ``tests/unit/replay/test_snapshot_schema.py`` is the
drift alarm: it validates real fixture snapshot lines
(``tests/fixtures/USR3SJiv.snapshots.jsonl``, produced by the Node side)
against the models below, so a shape change on the TypeScript side without
a matching update here fails that test on the next fixture regeneration.
It is not a guarantee of byte-for-byte sync at all times, only a check
against whatever fixtures happen to be committed.

Field names intentionally match ``types.ts``'s camelCase field names, not
Python's snake_case convention, to keep this file a straightforward visual
diff against its source of truth.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Intent internals are not modeled — see api_contract.py's docstring for
# the same reasoning: the full ~20-variant intent taxonomy is
# features/action_encoder.py's concern, not replay's.
StampedIntent = dict[str, object]


class SnapshotHeader(BaseModel):
    """The first line of every snapshot JSONL file, written once per game."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["header"]
    # 2: PlayerScalars gained `id`. Version-1 files fail validation here
    # rather than silently producing unresolvable player targets —
    # regenerate them with engine/bridge/src/replayGame.ts.
    formatVersion: Literal[2]
    gameId: str
    gitCommit: str
    map: str
    width: int
    height: int
    terrainGzipBase64: str
    snapshotEvery: int


class PlayerScalars(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The engine PlayerID: what intent target fields (targetID, recipient,
    # target, requestor) refer to — never clientID, even for humans. See
    # types.ts for the full explanation.
    id: str
    clientID: str | None
    smallID: int
    username: str
    type: str
    team: str | None
    isAlive: bool
    isTraitor: bool
    isDisconnected: bool
    numTilesOwned: int
    # Gold is a bigint in the engine; serialized as a decimal string.
    gold: str
    troops: float


class UnitSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    type: str
    ownerSmallID: int
    x: int
    y: int
    targetX: int | None
    targetY: int | None
    health: float
    maxHealth: float
    troops: float
    level: int
    isUnderConstruction: bool


class AttackSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    attackerSmallID: int
    # smallID of the defending player; 0 (TerraNullius) for unowned land.
    targetSmallID: int
    troops: float
    retreating: bool
    sourceX: int | None
    sourceY: int | None


class AllianceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requestorSmallID: int
    recipientSmallID: int
    expiresAtTick: int


class LegalityMask(BaseModel):
    """A coarse, player/unit-type-level legality mask — NOT a full
    tile-level mask. See ``types.ts``'s ``LegalityMask`` doc comment for
    why: tile-level legality for attack/build actions is derived by
    ``features/action_encoder.py`` by joining this against tile ownership,
    which it already has from the tile state.
    """

    model_config = ConfigDict(extra="forbid")

    attackablePlayerSmallIDs: list[int]
    canExpand: bool
    allianceRequestablePlayerSmallIDs: list[int]
    allianceAcceptablePlayerSmallIDs: list[int]
    breakableAlliancePlayerSmallIDs: list[int]
    donatableGoldPlayerSmallIDs: list[int]
    donatableTroopsPlayerSmallIDs: list[int]
    embargoablePlayerSmallIDs: list[int]
    targetablePlayerSmallIDs: list[int]
    buildableUnitTypes: list[str]
    canDeleteUnit: bool
    canEmbargoAll: bool


class Winner(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    id: str


class TickSnapshot(BaseModel):
    """One per-tick record, written every ``snapshotEvery`` ticks (outside
    the spawn phase). ``actions`` and ``legal`` are keyed by clientID and
    only present for currently-alive human players — nations/bots have no
    clientID and are never the imitation target.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["tick"]
    tick: int
    spawnPhase: bool
    tileStateGzipBase64: str
    players: list[PlayerScalars]
    units: list[UnitSnapshot]
    attacks: list[AttackSnapshot]
    alliances: list[AllianceSnapshot]
    # Intents each living human issued during [tick, tick+snapshotEvery).
    actions: dict[str, list[StampedIntent]]
    # Legality mask computed at this tick, before those actions were taken.
    legal: dict[str, LegalityMask]
    winner: Winner | None


SnapshotLine = Annotated[SnapshotHeader | TickSnapshot, Field(discriminator="kind")]
