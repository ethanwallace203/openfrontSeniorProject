"""Pydantic mirror of the raw OpenFront API's game record shape
(``GET /public/game/:gameId``), used to validate that a file landed in
``data/raw/`` correctly and is structurally complete.

This is a *sanity-check* contract, not an exhaustive one:
``engine/bridge/src/fetchGame.ts`` already validates the full response
against the engine's own canonical Zod ``GameRecordSchema`` before ever
writing the file to disk, so re-validating here is a belt-and-suspenders
check that the file wasn't truncated, corrupted, or hand-edited — not the
primary source of truth for the schema.

Field names intentionally match the raw JSON's camelCase, not Python's
snake_case convention, to keep this file a straightforward visual diff
against the API response and against
``engine/bridge/src/types.ts``/``Schemas.ts``. ``model_config = ConfigDict(extra="allow")``
throughout: this contract only pins down the fields ``orchestrator.py``
and downstream ``replay/`` code actually depend on existing — it
deliberately does not model the full intent-type taxonomy (union of ~20
variants), which is `src/openfront_il/features/action_encoder.py`'s
concern, not ingestion's.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class GameConfig(BaseModel):
    """A game's configuration, as sent by the lobby at game start."""

    model_config = ConfigDict(extra="allow")

    gameMap: str
    gameType: str
    gameMode: str
    gameMapSize: str
    bots: int
    randomSpawn: bool
    nations: str | int
    maxPlayers: int | None = None
    rankedType: str | None = None


class PlayerRecord(BaseModel):
    """One human player's record within a game."""

    model_config = ConfigDict(extra="allow")

    clientID: str
    username: str
    clanTag: str | None = None
    isLobbyCreator: bool | None = None
    teamIndex: int | None = None
    friends: list[str] | None = None


class Tribe(BaseModel):
    """A purchased bot tribe name in use for this game."""

    model_config = ConfigDict(extra="allow")

    name: str


class GameInfo(BaseModel):
    """The ``info`` block of a game record: metadata plus per-player and
    per-tribe records, but not the tick-by-tick turns."""

    model_config = ConfigDict(extra="allow")

    gameID: str
    config: GameConfig
    players: list[PlayerRecord]
    lobbyCreatedAt: int
    start: int
    end: int
    duration: int
    num_turns: int
    winner: list[object] | None = None
    tribes: list[Tribe] | None = None
    listed: bool | None = None


class Turn(BaseModel):
    """One tick's worth of intents, as recorded (sparse — a tick with no
    intents may be entirely absent from ``turns``, see
    ``src/openfront_il/replay/README.md``)."""

    model_config = ConfigDict(extra="allow")

    turnNumber: int
    # Intent internals are not modeled here — see this module's docstring.
    intents: list[dict[str, object]]
    hash: int | None = None


class GameRecord(BaseModel):
    """The full raw game record written to ``data/raw/{gameId}.json``."""

    model_config = ConfigDict(extra="allow")

    version: str
    gitCommit: str
    # Absent on at least Singleplayer records observed in practice — only
    # multiplayer games seem to carry a hosting subdomain/domain.
    domain: str | None = None
    subdomain: str | None = None
    info: GameInfo
    turns: list[Turn]
