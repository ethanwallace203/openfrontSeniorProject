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

from openfront_il.replay.snapshot_schema import SnapshotHeader, TickSnapshot, UnitSnapshot

# Terrain byte bit layout — engine/openfront-ai/openfront/src/core/game/GameMap.ts
_IS_LAND_BIT = 7
_SHORELINE_BIT = 6

# Tile state uint16 bit layout — same file
_PLAYER_ID_MASK = 0xFFF
_FALLOUT_BIT = 13
_DEFENSE_BONUS_BIT = 14

# TerraNullius's reserved "unowned land" sentinel smallID (engine convention).
TERRA_NULLIUS_SMALL_ID = 0

# Engine UnitType string values (Game.ts). Defined here rather than reusing
# action_encoder.UNIT_TYPES because action_encoder imports this module.
# Structures get one viewer-side channel each; trade ships, trains and
# shells are deliberately left out — they're constant background traffic,
# not decisions anyone reacts to.
STRUCTURE_UNIT_TYPES = {
    "City": "own_city",
    "Port": "own_port",
    "Factory": "own_factory",
    "Missile Silo": "own_missile_silo",
    "SAM Launcher": "own_sam_launcher",
    "Defense Post": "own_defense_post",
}
NUKE_UNIT_TYPES = frozenset({"Atom Bomb", "Hydrogen Bomb", "MIRV", "MIRV Warhead"})

# An owner counts as "stronger"/"weaker" than the viewer once their troop
# counts differ by more than this in log1p units (~1.65x). Bucketed into
# three 0/1 channels instead of one continuous channel so every spatial
# channel stays exactly 0 or 1 (see encode_spatial's uint8 note).
STRENGTH_MARGIN = 0.5

# Spatial tensor channel order — the single source of truth for what
# channel N in encode_spatial()'s output means. Never hardcode a channel
# index anywhere else; index into this list instead.
SPATIAL_CHANNELS = [
    # Terrain and ownership.
    "is_land",
    "is_shoreline",
    "owned_by_viewer",
    "owned_by_ally",
    "owned_by_enemy",
    "has_fallout",
    "has_defense_bonus",
    # Owner strength relative to the viewer — set on every tile of every
    # other living player (allies included), never on the viewer's own.
    "owner_stronger",
    "owner_similar",
    "owner_weaker",
    # Attacks in progress: territory of whoever is attacking the viewer, and
    # of whoever the viewer is attacking.
    "owner_attacking_viewer",
    "owner_attacked_by_viewer",
    # The viewer's own structures, one channel per type (keys of
    # STRUCTURE_UNIT_TYPES), at the unit's tile.
    "own_city",
    "own_port",
    "own_factory",
    "own_missile_silo",
    "own_sam_launcher",
    "own_defense_post",
    # Enemy structures (allies excluded); SAMs separately, since they decide
    # whether a nuke gets through.
    "enemy_structure",
    "enemy_sam_launcher",
    # Ships at their current tile.
    "own_transport",
    "enemy_transport",
    "own_warship",
    "enemy_warship",
    # Where enemy nukes in flight are going to land.
    "incoming_nuke_target",
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
    # Attacks in progress, summed over every attack touching the viewer.
    "incoming_attack_troops_log1p",
    "outgoing_attack_troops_log1p",
    "num_incoming_attacks",
    # Threats in transit: enemy boats (possible invasions) and enemy nukes.
    "enemy_transports_log1p",
    "enemy_nukes_in_flight",
    # What the viewer has built — decides what's worth building next.
    "own_city_count_log1p",
    "own_port_count_log1p",
    "own_factory_count_log1p",
    "own_missile_silo_count_log1p",
    "own_sam_launcher_count_log1p",
    "own_defense_post_count_log1p",
    "own_warship_count_log1p",
]

# Per-candidate feature order for the target_player pointer head — same
# source-of-truth convention. Every feature is relative to the viewer
# (ally/enemy, troop advantage, shared border), never a raw player ID,
# for the same reason as the spatial channels: an ID means nothing across
# games, but "my largest neighbor, who is attacking me" does.
PLAYER_FEATURES = [
    "is_terra_nullius",
    "is_ally",
    "is_human",
    "is_traitor",
    "tiles_owned_log1p",
    "troops_log1p",
    "gold_log1p",
    # log1p(viewer troops) - log1p(candidate troops): > 0 means the viewer
    # outnumbers them.
    "troop_advantage_log_ratio",
    # Count of 4-neighbor tile edges between the viewer's territory and
    # this candidate's (land only, so open ocean never counts as a border
    # with terra nullius).
    "shared_border_log1p",
    "is_attacking_viewer",
    "viewer_is_attacking",
    # How hard, not just whether: troops committed to attacks between this
    # candidate and the viewer, in each direction.
    "incoming_attack_troops_log1p",
    "outgoing_attack_troops_log1p",
    # Troop density — how hard their territory is to push into.
    "troops_per_tile_log1p",
    # Where they are: the center of their land minus the center of the
    # viewer's, as a fraction of map width/height. For terra nullius, the
    # center of all unowned land. This is what lets boats pick targets the
    # viewer doesn't border.
    "centroid_dx",
    "centroid_dy",
    "centroid_distance",
    "is_nation",
    "is_bot",
    # Worth taking / safe to nuke.
    "structure_count_log1p",
    "has_sam_launcher",
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


def _attack_troops(
    tick: TickSnapshot, viewer_small_id: int
) -> tuple[dict[int, float], dict[int, float]]:
    """``(incoming, outgoing)``: troops in attacks on the viewer, keyed by
    attacker smallID, and troops in the viewer's attacks, keyed by target
    smallID (``TERRA_NULLIUS_SMALL_ID`` for expansion into unowned land)."""
    incoming: dict[int, float] = {}
    outgoing: dict[int, float] = {}
    for attack in tick.attacks:
        if attack.targetSmallID == viewer_small_id:
            attacker = attack.attackerSmallID
            incoming[attacker] = incoming.get(attacker, 0.0) + attack.troops
        elif attack.attackerSmallID == viewer_small_id:
            target = attack.targetSmallID
            outgoing[target] = outgoing.get(target, 0.0) + attack.troops
    return incoming, outgoing


# Channel N of the spatial tensor is bit N of the packed int32 encoding
# (encode_spatial_packed). int32 rather than uint32 because torch's bit
# shifts are defined for signed ints; 25 channels leave the sign bit unused.
assert len(SPATIAL_CHANNELS) <= 31, "packed spatial encoding holds at most 31 channels"
_BIT = {name: np.int32(1 << index) for index, name in enumerate(SPATIAL_CHANNELS)}

# Terrain never changes within a game, so its two channels are computed
# once per game and reused for every tick and viewer. Keyed by gameId; a
# handful of entries covers every game a DataLoader worker interleaves.
_TERRAIN_BITS_CACHE: dict[str, np.ndarray] = {}
_TERRAIN_BITS_CACHE_SIZE = 8


def _terrain_bits(header: SnapshotHeader) -> np.ndarray:
    """``int32`` ``(H, W)``: the ``is_land`` and ``is_shoreline`` bits."""
    cached = _TERRAIN_BITS_CACHE.get(header.gameId)
    if cached is not None and cached.shape == (header.height, header.width):
        return cached
    terrain = decode_terrain(header).astype(np.int32)
    bits: np.ndarray = ((terrain >> _IS_LAND_BIT) & 1) * _BIT["is_land"]
    bits |= ((terrain >> _SHORELINE_BIT) & 1) * _BIT["is_shoreline"]
    if len(_TERRAIN_BITS_CACHE) >= _TERRAIN_BITS_CACHE_SIZE:
        _TERRAIN_BITS_CACHE.pop(next(iter(_TERRAIN_BITS_CACHE)))
    _TERRAIN_BITS_CACHE[header.gameId] = bits
    return bits


def _set_unit_bits(
    packed: np.ndarray, units: list[UnitSnapshot], bit: np.int32, *, at_target: bool = False
) -> None:
    """ORs ``bit`` into ``packed`` at each unit's tile (or its target tile,
    for ``at_target``). Units without a target are skipped when
    ``at_target``, and anything off the map is ignored rather than raising."""
    height, width = packed.shape
    for unit in units:
        x, y = (unit.targetX, unit.targetY) if at_target else (unit.x, unit.y)
        if x is not None and y is not None and 0 <= x < width and 0 <= y < height:
            packed[y, x] |= bit


def _strength_buckets(
    tick: TickSnapshot, viewer_small_id: int
) -> tuple[set[int], set[int], set[int]]:
    """``(stronger, similar, weaker)`` smallIDs of every other living player,
    by troop count relative to the viewer (see ``STRENGTH_MARGIN``)."""
    viewer = next((p for p in tick.players if p.smallID == viewer_small_id), None)
    viewer_troops_log = np.log1p(viewer.troops) if viewer is not None else 0.0
    stronger: set[int] = set()
    similar: set[int] = set()
    weaker: set[int] = set()
    for player in tick.players:
        if player.smallID == viewer_small_id or not player.isAlive:
            continue
        advantage = viewer_troops_log - np.log1p(player.troops)
        if advantage < -STRENGTH_MARGIN:
            stronger.add(player.smallID)
        elif advantage > STRENGTH_MARGIN:
            weaker.add(player.smallID)
        else:
            similar.add(player.smallID)
    return stronger, similar, weaker


def encode_spatial_packed(
    header: SnapshotHeader, tick: TickSnapshot, viewer_small_id: int
) -> np.ndarray:
    """The spatial observation, bit-packed: ``int32`` ``(height, width)``
    where bit N of a tile is that tile's value in channel
    ``SPATIAL_CHANNELS[N]`` (every channel is 0 or 1). ``encode_spatial``
    is exactly this, unpacked.

    Built directly in packed form because that's what keeps training data
    loading fast: a full ``[C, H, W]`` tensor is 25 separate full-map
    arrays, and building + stacking + copying them out of DataLoader
    workers was the training bottleneck (memory bandwidth, not compute).
    Here it's a handful of full-map passes:

    - terrain bits: computed once per game and cached (``_terrain_bits``);
    - all 10 owner-derived channels (ownership, strength, attacks): one
      lookup table indexed by owner smallID, applied in a single gather;
    - fallout / defense bonus: two shifts of the tile state;
    - units: set only at the tiles they occupy.
    """
    tile_state = decode_tile_state(header, tick).astype(np.int32)
    owner_id = tile_state & _PLAYER_ID_MASK
    allies = _allied_small_ids(tick, viewer_small_id)
    stronger, similar, weaker = _strength_buckets(tick, viewer_small_id)
    incoming, outgoing = _attack_troops(tick, viewer_small_id)

    def in_range(small_ids: set[int]) -> list[int]:
        return [i for i in small_ids if 0 <= i <= _PLAYER_ID_MASK]

    # One entry per possible owner: every channel bit that owner's tiles get.
    owner_bits = np.zeros(_PLAYER_ID_MASK + 1, dtype=np.int32)
    owner_bits[1:] = _BIT["owned_by_enemy"]  # any owner but terra nullius...
    owner_bits[in_range(allies)] = _BIT["owned_by_ally"]  # ...unless an ally
    if 0 <= viewer_small_id <= _PLAYER_ID_MASK:
        owner_bits[viewer_small_id] = _BIT["owned_by_viewer"]  # ...or the viewer
    owner_bits[TERRA_NULLIUS_SMALL_ID] = 0
    owner_bits[in_range(stronger)] |= _BIT["owner_stronger"]
    owner_bits[in_range(similar)] |= _BIT["owner_similar"]
    owner_bits[in_range(weaker)] |= _BIT["owner_weaker"]
    owner_bits[in_range(set(incoming))] |= _BIT["owner_attacking_viewer"]
    # Expansion into terra nullius is an outgoing attack too, but unowned
    # land isn't anyone's territory to mark.
    owner_bits[in_range(set(outgoing) - {TERRA_NULLIUS_SMALL_ID})] |= _BIT[
        "owner_attacked_by_viewer"
    ]

    packed: np.ndarray = owner_bits[owner_id]
    packed |= _terrain_bits(header)
    packed |= ((tile_state >> _FALLOUT_BIT) & 1) * _BIT["has_fallout"]
    packed |= ((tile_state >> _DEFENSE_BONUS_BIT) & 1) * _BIT["has_defense_bonus"]

    mine = [u for u in tick.units if u.ownerSmallID == viewer_small_id]
    enemy = [
        u
        for u in tick.units
        if u.ownerSmallID != viewer_small_id
        and u.ownerSmallID != TERRA_NULLIUS_SMALL_ID
        and u.ownerSmallID not in allies
    ]
    for unit_type, channel_name in STRUCTURE_UNIT_TYPES.items():
        _set_unit_bits(packed, [u for u in mine if u.type == unit_type], _BIT[channel_name])
    _set_unit_bits(
        packed, [u for u in enemy if u.type in STRUCTURE_UNIT_TYPES], _BIT["enemy_structure"]
    )
    _set_unit_bits(
        packed, [u for u in enemy if u.type == "SAM Launcher"], _BIT["enemy_sam_launcher"]
    )
    _set_unit_bits(packed, [u for u in mine if u.type == "Transport"], _BIT["own_transport"])
    _set_unit_bits(packed, [u for u in enemy if u.type == "Transport"], _BIT["enemy_transport"])
    _set_unit_bits(packed, [u for u in mine if u.type == "Warship"], _BIT["own_warship"])
    _set_unit_bits(packed, [u for u in enemy if u.type == "Warship"], _BIT["enemy_warship"])
    _set_unit_bits(
        packed,
        [u for u in enemy if u.type in NUKE_UNIT_TYPES],
        _BIT["incoming_nuke_target"],
        at_target=True,
    )
    return packed


def unpack_spatial(packed: np.ndarray) -> np.ndarray:
    """Packed ``(H, W)`` ``int32`` -> ``(len(SPATIAL_CHANNELS), H, W)``
    ``uint8`` 0/1 — the full spatial tensor.

    Works one byte-plane at a time (bits 0-7, 8-15, ...) in ``uint8`` —
    measured ~2x faster than shifting the full ``int32`` once per channel,
    since every pass touches a quarter of the memory."""
    count = len(SPATIAL_CHANNELS)
    as_bytes = packed.astype("<i4", copy=False).view(np.uint8).reshape(*packed.shape, 4)
    channels = np.empty((count, *packed.shape), dtype=np.uint8)
    for byte in range((count + 7) // 8):
        plane = np.ascontiguousarray(as_bytes[..., byte])
        for bit in range(min(8, count - byte * 8)):
            channels[byte * 8 + bit] = (plane >> bit) & 1
    return channels


def pack_spatial(spatial: np.ndarray) -> np.ndarray:
    """Inverse of ``unpack_spatial``: ``(C, H, W)`` 0/1 -> packed ``(H, W)``
    ``int32``. For tests and tools; the encoders build packed form directly."""
    packed = np.zeros(spatial.shape[1:], dtype=np.int32)
    for index in range(spatial.shape[0]):
        packed |= spatial[index].astype(np.int32) << index
    return packed


def spatial_channel(spatial: np.ndarray, name: str) -> np.ndarray:
    """One channel of a spatial observation as an ``(H, W)`` bool mask,
    from either form — full ``(C, H, W)`` or packed ``(H, W)``. Lets code
    that only needs a few channels (e.g. the tile-target masks) read them
    straight from packed form without unpacking all 25."""
    index = SPATIAL_CHANNELS.index(name)
    if spatial.ndim == 2:
        channel: np.ndarray = ((spatial >> index) & 1).astype(bool)
    else:
        channel = np.asarray(spatial[index] > 0)
    return channel


def encode_spatial(header: SnapshotHeader, tick: TickSnapshot, viewer_small_id: int) -> np.ndarray:
    """Builds the ``(len(SPATIAL_CHANNELS), height, width)`` spatial
    tensor for one tick, from ``viewer_small_id``'s perspective —
    ``encode_spatial_packed``, unpacked (that function documents every
    channel's rule).

    ``uint8``, not float: every channel is exactly 0 or 1, so one byte per
    value loses nothing and is 4x smaller than float32. ``SpatialEncoder``
    casts to float itself, on whatever device it runs on.
    """
    return unpack_spatial(encode_spatial_packed(header, tick, viewer_small_id))


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

    incoming, outgoing = _attack_troops(tick, viewer_small_id)
    own_counts: dict[str, int] = {}
    enemy_transports = 0
    enemy_nukes_in_flight = 0
    for unit in tick.units:
        if unit.ownerSmallID == viewer_small_id:
            own_counts[unit.type] = own_counts.get(unit.type, 0) + 1
        elif unit.ownerSmallID != TERRA_NULLIUS_SMALL_ID and unit.ownerSmallID not in allies:
            if unit.type == "Transport":
                enemy_transports += 1
            elif unit.type in NUKE_UNIT_TYPES and unit.targetX is not None:
                enemy_nukes_in_flight += 1

    def own_count_log(unit_type: str) -> float:
        return float(np.log1p(own_counts.get(unit_type, 0)))

    values: dict[str, float] = {
        "viewer_gold_log1p": np.log1p(float(viewer.gold)),
        "viewer_troops_log1p": np.log1p(viewer.troops),
        "viewer_num_tiles_owned_log1p": np.log1p(viewer.numTilesOwned),
        "viewer_is_alive": float(viewer.isAlive),
        "viewer_is_traitor": float(viewer.isTraitor),
        "viewer_num_alliances": len(allies),
        "viewer_units_in_flight": units_in_flight,
        "enemy_total_tiles_owned_log1p": np.log1p(enemy_total_tiles),
        "enemy_total_troops_log1p": np.log1p(enemy_total_troops),
        "enemy_alive_count": enemy_alive_count,
        "ally_alive_count": ally_alive_count,
        "spawn_phase": float(tick.spawnPhase),
        "incoming_attack_troops_log1p": np.log1p(sum(incoming.values())),
        "outgoing_attack_troops_log1p": np.log1p(sum(outgoing.values())),
        "num_incoming_attacks": sum(
            1 for attack in tick.attacks if attack.targetSmallID == viewer_small_id
        ),
        "enemy_transports_log1p": np.log1p(enemy_transports),
        "enemy_nukes_in_flight": enemy_nukes_in_flight,
        "own_city_count_log1p": own_count_log("City"),
        "own_port_count_log1p": own_count_log("Port"),
        "own_factory_count_log1p": own_count_log("Factory"),
        "own_missile_silo_count_log1p": own_count_log("Missile Silo"),
        "own_sam_launcher_count_log1p": own_count_log("SAM Launcher"),
        "own_defense_post_count_log1p": own_count_log("Defense Post"),
        "own_warship_count_log1p": own_count_log("Warship"),
    }
    # Ordered by FLAT_FEATURES (a KeyError means a feature was named but
    # never computed), so columns can never silently shift.
    return np.array([values[name] for name in FLAT_FEATURES], dtype=np.float32)


def _border_edge_counts(
    owner_id: np.ndarray, is_land: np.ndarray, viewer_small_id: int
) -> np.ndarray:
    """``(len(owner smallID space),)`` count of 4-neighbor edges from a
    viewer-owned tile to a land tile owned by each smallID."""
    mine = owner_id == viewer_small_id
    counts = np.zeros(_PLAYER_ID_MASK + 1, dtype=np.int64)
    # (from, to) slice pairs covering each of the four neighbor directions.
    shifts = [
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
    ]
    for source, neighbor in shifts:
        crossing = mine[source] & ~mine[neighbor] & is_land[neighbor]
        counts += np.bincount(owner_id[neighbor][crossing], minlength=len(counts))
    return counts


def _land_centroids(
    owner_id: np.ndarray, is_land: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(center_x, center_y, has_land)``, each indexed by smallID: the mean
    tile coordinate of every land tile each owner holds (terra nullius
    included — the center of unowned land). One ``bincount`` pass per axis
    for every player at once."""
    rows, columns = np.nonzero(is_land)
    owners = owner_id[rows, columns]
    size = _PLAYER_ID_MASK + 1
    counts = np.bincount(owners, minlength=size)
    safe = np.maximum(counts, 1)
    center_x = np.bincount(owners, weights=columns, minlength=size) / safe
    center_y = np.bincount(owners, weights=rows, minlength=size) / safe
    return center_x, center_y, counts > 0


def encode_player_candidates(
    header: SnapshotHeader, tick: TickSnapshot, viewer_small_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Builds the candidate set for the ``target_player`` head, from
    ``viewer_small_id``'s perspective.

    Candidates are terra nullius first (so "attack unowned land" — the
    most common attack in the game — is a target like any other), then
    every other living player in ascending smallID order. The viewer is
    never a candidate.

    Returns:
        ``(small_ids [N] int64, features [N, len(PLAYER_FEATURES)]
        float32)``.

    Raises:
        ValueError: if no player in this tick has ``viewer_small_id``.
    """
    viewer = next((p for p in tick.players if p.smallID == viewer_small_id), None)
    if viewer is None:
        raise ValueError(f"no player with smallID {viewer_small_id} in this tick")

    is_land = ((decode_terrain(header) >> _IS_LAND_BIT) & 1).astype(bool)
    owner_id = decode_tile_state(header, tick) & _PLAYER_ID_MASK
    border = _border_edge_counts(owner_id, is_land, viewer_small_id)
    allies = _allied_small_ids(tick, viewer_small_id)
    incoming, outgoing = _attack_troops(tick, viewer_small_id)
    viewer_troops_log = np.log1p(viewer.troops)
    center_x, center_y, has_land = _land_centroids(owner_id, is_land)

    structures: dict[int, int] = {}
    has_sam: set[int] = set()
    for unit in tick.units:
        if unit.type in STRUCTURE_UNIT_TYPES:
            structures[unit.ownerSmallID] = structures.get(unit.ownerSmallID, 0) + 1
        if unit.type == "SAM Launcher":
            has_sam.add(unit.ownerSmallID)

    def location(small_id: int) -> dict[str, float]:
        # Relative to the viewer's own land; zero if either side owns none
        # (e.g. the viewer just died, or no unowned land is left).
        if not (has_land[small_id] and has_land[viewer_small_id]):
            return {}
        dx = (center_x[small_id] - center_x[viewer_small_id]) / header.width
        dy = (center_y[small_id] - center_y[viewer_small_id]) / header.height
        return {"centroid_dx": dx, "centroid_dy": dy, "centroid_distance": float(np.hypot(dx, dy))}

    unowned_land = int((is_land & (owner_id == TERRA_NULLIUS_SMALL_ID)).sum())
    small_ids = [TERRA_NULLIUS_SMALL_ID]
    rows: list[dict[str, float]] = [
        {
            "is_terra_nullius": 1.0,
            "tiles_owned_log1p": np.log1p(unowned_land),
            # Unowned land has no troops, so the viewer's whole army is the
            # advantage.
            "troop_advantage_log_ratio": viewer_troops_log,
            "shared_border_log1p": np.log1p(border[TERRA_NULLIUS_SMALL_ID]),
            "viewer_is_attacking": float(TERRA_NULLIUS_SMALL_ID in outgoing),
            "outgoing_attack_troops_log1p": np.log1p(outgoing.get(TERRA_NULLIUS_SMALL_ID, 0.0)),
            **location(TERRA_NULLIUS_SMALL_ID),
        }
    ]
    others = sorted(
        (p for p in tick.players if p.smallID != viewer_small_id and p.isAlive),
        key=lambda p: p.smallID,
    )
    for player in others:
        small_ids.append(player.smallID)
        rows.append(
            {
                "is_ally": float(player.smallID in allies),
                "is_human": float(player.clientID is not None),
                "is_traitor": float(player.isTraitor),
                "tiles_owned_log1p": np.log1p(player.numTilesOwned),
                "troops_log1p": np.log1p(player.troops),
                "gold_log1p": np.log1p(float(player.gold)),
                "troop_advantage_log_ratio": viewer_troops_log - np.log1p(player.troops),
                "shared_border_log1p": np.log1p(border[player.smallID]),
                "is_attacking_viewer": float(player.smallID in incoming),
                "viewer_is_attacking": float(player.smallID in outgoing),
                "incoming_attack_troops_log1p": np.log1p(incoming.get(player.smallID, 0.0)),
                "outgoing_attack_troops_log1p": np.log1p(outgoing.get(player.smallID, 0.0)),
                "troops_per_tile_log1p": np.log1p(player.troops / max(player.numTilesOwned, 1)),
                "is_nation": float(player.type == "NATION"),
                "is_bot": float(player.type == "BOT"),
                "structure_count_log1p": np.log1p(structures.get(player.smallID, 0)),
                "has_sam_launcher": float(player.smallID in has_sam),
                **location(player.smallID),
            }
        )
    # Each row is a dict ordered by PLAYER_FEATURES; anything a row doesn't
    # set (e.g. troops for terra nullius) is 0. Building rows by name rather
    # than by position is what keeps a new feature from silently shifting
    # every column after it.
    features = [[row.get(name, 0.0) for name in PLAYER_FEATURES] for row in rows]
    return np.array(small_ids, dtype=np.int64), np.array(features, dtype=np.float32)


def encode_candidate_cells(
    header: SnapshotHeader,
    tick: TickSnapshot,
    viewer_small_id: int,
    candidate_small_ids: np.ndarray,
    cell_factor: int,
) -> np.ndarray:
    """``int16`` ``[H // cell_factor, W // cell_factor]``: for each cell of
    ``SpatialEncoder``'s feature map, the index (into ``candidate_small_ids``,
    as returned by ``encode_player_candidates``) of whoever owns most of the
    cell's land — ``-1`` if the cell has no land owned by a candidate.

    This is what lets ``PolicyNetwork`` describe each target-player
    candidate by *what their territory looks like* (the feature map averaged
    over their cells), not only by their scalar row. Water and the viewer's
    own tiles are ignored when picking a cell's owner, so a coastal or
    front-line cell belongs to the neighbor or unowned land in it, not to
    the sea or the viewer. Terra nullius is candidate 0, so unclaimed land
    gets cells like any player.

    ``cell_factor`` must match the model's ``SpatialEncoder.downsample_factor``;
    cells floor-divide exactly like the conv trunk's pooling, so a map's last
    ``H % cell_factor`` rows (and columns) are dropped the same way.
    """
    cell_height, cell_width = header.height // cell_factor, header.width // cell_factor
    if cell_height == 0 or cell_width == 0:
        return np.full((cell_height, cell_width), -1, dtype=np.int16)

    is_land = ((decode_terrain(header) >> _IS_LAND_BIT) & 1).astype(bool)
    owner_id = (decode_tile_state(header, tick) & _PLAYER_ID_MASK).astype(np.int32)
    # Anything that shouldn't decide a cell's owner becomes a sentinel
    # above every real smallID, so it sorts last and can be skipped.
    ignored = _PLAYER_ID_MASK + 1
    owner_id[~is_land | (owner_id == viewer_small_id)] = ignored

    rows, columns = cell_height * cell_factor, cell_width * cell_factor
    blocks = (
        owner_id[:rows, :columns]
        .reshape(cell_height, cell_factor, cell_width, cell_factor)
        .transpose(0, 2, 1, 3)
        .reshape(cell_height * cell_width, cell_factor * cell_factor)
    )
    majority = _row_mode(blocks, ignored)

    to_candidate = np.full(ignored + 1, -1, dtype=np.int16)
    to_candidate[candidate_small_ids] = np.arange(len(candidate_small_ids), dtype=np.int16)
    cells: np.ndarray = to_candidate[majority].reshape(cell_height, cell_width)
    return cells


def _row_mode(values: np.ndarray, ignored: int) -> np.ndarray:
    """The most common value in each row of ``values``, skipping
    ``ignored`` (a row of only ``ignored`` returns ``ignored``). Vectorized:
    sort each row, then the mode is where the longest run of equal values
    ends."""
    ordered = np.sort(values, axis=1)
    positions = np.arange(ordered.shape[1])
    run_starts = np.ones_like(ordered, dtype=bool)
    run_starts[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    start_index = np.maximum.accumulate(np.where(run_starts, positions, 0), axis=1)
    run_length = positions - start_index + 1
    run_length[ordered == ignored] = 0
    best = run_length.argmax(axis=1)
    mode: np.ndarray = ordered[np.arange(len(ordered)), best]
    mode[run_length.max(axis=1) == 0] = ignored
    return mode


def encode_observation(
    header: SnapshotHeader, tick: TickSnapshot, viewer_small_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(header, tick, viewer) -> (spatial [C,H,W], flat [F])``."""
    return (
        encode_spatial(header, tick, viewer_small_id),
        encode_flat(tick, viewer_small_id),
    )
