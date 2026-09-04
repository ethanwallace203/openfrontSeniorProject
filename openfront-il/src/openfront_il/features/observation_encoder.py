"""Pure functions: ``(header, tick snapshot, viewing player) -> (spatial
tensor, flat vector)``.

No I/O side effects — ``dataset_builder.py`` owns reading
``data/snapshots/``. Every function here is a pure transform over
already-parsed ``snapshot_schema`` objects, so it's testable with a
hand-constructed ``TickSnapshot`` and no disk access.

Design principle (from prior art, see ``features/README.md``): compress
only the large, high-dimensional tile grid through a learned spatial
encoder (``models/spatial_encoder.py``); every small, exact fact here —
per-player scalars, diplomacy, units in flight — passes straight through
as the flat vector, uncompressed.

Every tensor is encoded **from one specific player's perspective**
(``viewer_small_id``): "owned by me" vs. "owned by an ally" vs. "owned by
an enemy," not raw, game-arbitrary owner IDs. A raw owner ID has no fixed
meaning across different games or even across different players within
the same game, so a perspective-free encoding would give the policy
nothing generalizable to learn from.
"""

from __future__ import annotations

import base64
import gzip

import numpy as np

from openfront_il.replay.snapshot_schema import SnapshotHeader, TickSnapshot

# Terrain byte bit layout — engine/openfront-ai/openfront/src/core/game/GameMap.ts
_IS_LAND_BIT = 7
_SHORELINE_BIT = 6

# Tile state uint16 bit layout — same file
_PLAYER_ID_MASK = 0xFFF
_FALLOUT_BIT = 13
_DEFENSE_BONUS_BIT = 14

# TerraNullius's reserved "unowned land" sentinel smallID (engine convention).
_TERRA_NULLIUS_SMALL_ID = 0

# Spatial tensor channel order — the single source of truth for what
# channel N in encode_spatial()'s output means. Never hardcode a channel
# index anywhere else; index into this list instead.
SPATIAL_CHANNELS = [
    "is_land",
    "is_shoreline",
    "owned_by_viewer",
    "owned_by_ally",
    "owned_by_enemy",
    "has_fallout",
    "has_defense_bonus",
]

# Flat feature order — same source-of-truth convention.
#
# `_log1p` fields: real games have troop/gold/tile counts ranging from 0
# to the tens of thousands (measured — see training/README.md's "CPU
# training on real maps" note, where feeding these raw into a linear
# layer produced logits large enough to blow up cross-entropy loss into
# the hundreds of thousands on the very first epoch). log1p keeps these
# on a comparable, network-friendly scale to the naturally-small counts
# (alliances, alive-player counts) below, which are left raw.
FLAT_FEATURES = [
    "viewer_gold_log1p",
    "viewer_troops_log1p",
    "viewer_num_tiles_owned_log1p",
    "viewer_is_alive",
    "viewer_is_traitor",
    "viewer_num_alliances",
    "viewer_units_in_flight",
    "enemy_total_tiles_owned_log1p",
    "enemy_total_troops_log1p",
    "enemy_alive_count",
    "ally_alive_count",
    "spawn_phase",
]


def decode_terrain(header: SnapshotHeader) -> np.ndarray:
    """Decodes the gzip+base64 terrain buffer into a ``(height, width)``
    ``uint8`` array. Terrain is written once per game (it never changes
    tick to tick) — see ``SnapshotHeader``.
    """
    raw = gzip.decompress(base64.b64decode(header.terrainGzipBase64))
    return np.frombuffer(raw, dtype=np.uint8).reshape(header.height, header.width)


def decode_tile_state(header: SnapshotHeader, tick: TickSnapshot) -> np.ndarray:
    """Decodes one tick's gzip+base64 tile-state buffer into a
    ``(height, width)`` ``uint16`` array (little-endian, matching
    ``game.tileStateBuffer()``'s own byte order).
    """
    raw = gzip.decompress(base64.b64decode(tick.tileStateGzipBase64))
    return np.frombuffer(raw, dtype="<u2").reshape(header.height, header.width)


def _allied_small_ids(tick: TickSnapshot, viewer_small_id: int) -> set[int]:
    allies: set[int] = set()
    for alliance in tick.alliances:
        if alliance.requestorSmallID == viewer_small_id:
            allies.add(alliance.recipientSmallID)
        elif alliance.recipientSmallID == viewer_small_id:
            allies.add(alliance.requestorSmallID)
    return allies


def encode_spatial(header: SnapshotHeader, tick: TickSnapshot, viewer_small_id: int) -> np.ndarray:
    """Builds the ``(len(SPATIAL_CHANNELS), height, width)`` spatial
    tensor for one tick, from ``viewer_small_id``'s perspective.
    """
    terrain = decode_terrain(header)
    tile_state = decode_tile_state(header, tick)

    is_land = ((terrain >> _IS_LAND_BIT) & 1).astype(np.float32)
    is_shoreline = ((terrain >> _SHORELINE_BIT) & 1).astype(np.float32)

    owner_id = tile_state & _PLAYER_ID_MASK
    allies = _allied_small_ids(tick, viewer_small_id)
    ally_list = list(allies)

    owned_by_viewer = (owner_id == viewer_small_id).astype(np.float32)
    owned_by_ally = np.isin(owner_id, ally_list).astype(np.float32)
    is_owned = owner_id != _TERRA_NULLIUS_SMALL_ID
    owned_by_enemy = (
        is_owned & (owner_id != viewer_small_id) & ~np.isin(owner_id, ally_list)
    ).astype(np.float32)

    has_fallout = ((tile_state >> _FALLOUT_BIT) & 1).astype(np.float32)
    has_defense_bonus = ((tile_state >> _DEFENSE_BONUS_BIT) & 1).astype(np.float32)

    return np.stack(
        [
            is_land,
            is_shoreline,
            owned_by_viewer,
            owned_by_ally,
            owned_by_enemy,
            has_fallout,
            has_defense_bonus,
        ],
        axis=0,
    )


def encode_flat(tick: TickSnapshot, viewer_small_id: int) -> np.ndarray:
    """Builds the ``(len(FLAT_FEATURES),)`` flat feature vector for one
    tick, from ``viewer_small_id``'s perspective.

    Raises:
        ValueError: if no player in this tick has ``viewer_small_id``.
    """
    viewer = next((p for p in tick.players if p.smallID == viewer_small_id), None)
    if viewer is None:
        raise ValueError(f"no player with smallID {viewer_small_id} in this tick")

    allies = _allied_small_ids(tick, viewer_small_id)

    enemy_total_tiles = 0
    enemy_total_troops = 0.0
    enemy_alive_count = 0
    ally_alive_count = 0
    for player in tick.players:
        if player.smallID == viewer_small_id or not player.isAlive:
            continue
        if player.smallID in allies:
            ally_alive_count += 1
            continue
        enemy_alive_count += 1
        enemy_total_tiles += player.numTilesOwned
        enemy_total_troops += player.troops

    units_in_flight = sum(
        1
        for unit in tick.units
        if unit.ownerSmallID == viewer_small_id and unit.targetX is not None
    )

    return np.array(
        [
            np.log1p(float(viewer.gold)),
            np.log1p(viewer.troops),
            np.log1p(viewer.numTilesOwned),
            float(viewer.isAlive),
            float(viewer.isTraitor),
            len(allies),
            units_in_flight,
            np.log1p(enemy_total_tiles),
            np.log1p(enemy_total_troops),
            enemy_alive_count,
            ally_alive_count,
            float(tick.spawnPhase),
        ],
        dtype=np.float32,
    )


def encode_observation(
    header: SnapshotHeader, tick: TickSnapshot, viewer_small_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(header, tick, viewer) -> (spatial [C,H,W], flat [F])``."""
    return (
        encode_spatial(header, tick, viewer_small_id),
        encode_flat(tick, viewer_small_id),
    )
