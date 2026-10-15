"""Pure functions: engine action -> multi-head discrete action label(s) +
legality masks.

See ``docs/imitation_learning_plan.md`` Section 2 for the full taxonomy
design. Tile and player targets are stored raw here (``target_x``/
``target_y``, ``target_player_small_id``), not one-hot encoded: a
fixed-size classifier over every tile or over a per-game player roster
isn't tractable. Instead they're consumed by ``PolicyNetwork``'s
pointer-style ``tile_target`` / ``target_player`` heads, which score a
variable number of cells / candidates — training converts the raw fields
to a cell index / candidate index at load time.

Which action types take which argument is defined once, here, by
``TILE_TARGET_ACTION_TYPES`` / ``PLAYER_TARGET_ACTION_TYPES``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openfront_il.features.observation_encoder import SPATIAL_CHANNELS, TERRA_NULLIUS_SMALL_ID
from openfront_il.replay.snapshot_schema import LegalityMask

# The modeled action_type head. "noop" is not a real intent type — it's
# assigned by dataset_builder.py when a player issues no intent in a
# snapshot window. See docs/imitation_learning_plan.md Section 2 for which
# real intent types were deliberately excluded from this list.
ACTION_TYPES = [
    "noop",
    "spawn",
    "attack",
    "boat",
    "build_unit",
    "upgrade_structure",
    "move_warship",
    "cancel_boat",
    "cancel_attack",
    "delete_unit",
    "allianceRequest",
    "allianceReject",
    "allianceExtension",
    "breakAlliance",
    "donate_gold",
    "donate_troops",
    "embargo",
    "embargo_all",
    "targetPlayer",
]
_ACTION_TYPE_INDEX = {name: index for index, name in enumerate(ACTION_TYPES)}

# The secondary unit_type head, used only when action_type is build_unit or
# upgrade_structure. Values match the engine's UnitType enum's *string*
# values exactly (Game.ts), not its enum key names.
UNIT_TYPES = [
    "none",
    "Transport",
    "Warship",
    "Shell",
    "SAMMissile",
    "Port",
    "Atom Bomb",
    "Hydrogen Bomb",
    "Trade Ship",
    "Missile Silo",
    "Defense Post",
    "SAM Launcher",
    "City",
    "MIRV",
    "MIRV Warhead",
    "Train",
    "Factory",
]
_UNIT_TYPE_INDEX = {name: index for index, name in enumerate(UNIT_TYPES)}

# Action types whose intent carries a TileRef the tile_target head
# predicts (spawn/build_unit/move_warship: `tile`; boat: `dst`).
TILE_TARGET_ACTION_TYPES = frozenset({"spawn", "boat", "build_unit", "move_warship"})

# Action types whose intent names a player the target_player head
# predicts. `attack` with a null targetID means "expand into unowned
# land" and is labeled as targeting terra nullius, not skipped.
PLAYER_TARGET_ACTION_TYPES = frozenset(
    {
        "attack",
        "allianceRequest",
        "allianceReject",
        "allianceExtension",
        "breakAlliance",
        "donate_gold",
        "donate_troops",
        "embargo",
        "targetPlayer",
    }
)

# build_unit's `tile` is the nuke's *target* for these, so anywhere is a
# valid cell — not just the builder's own territory.
_NUKE_UNIT_TYPES = frozenset({"Atom Bomb", "Hydrogen Bomb", "MIRV"})

# Real intent types excluded from the modeled action space entirely (see
# docs/imitation_learning_plan.md Section 2). An intent of one of these
# types should be skipped by dataset_builder.py, not encoded as noop.
UNMODELED_INTENT_TYPES = frozenset(
    {
        "emoji",
        "quick_chat",
        "mark_disconnected",
        "kick_player",
        "toggle_pause",
        "update_game_config",
        "toggle_game_start_timer",
    }
)


@dataclass(frozen=True)
class EncodedAction:
    """One (multi-head) encoded action label. `target_x`/`target_y`/
    `target_player_small_id`/`amount` are raw, not one-hot encoded — see
    this module's docstring for why."""

    action_type: int
    unit_type: int
    target_x: int | None = None
    target_y: int | None = None
    target_player_small_id: int | None = None
    amount: float | None = None


NOOP_ACTION = EncodedAction(
    action_type=_ACTION_TYPE_INDEX["noop"], unit_type=_UNIT_TYPE_INDEX["none"]
)


def _as_float(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _resolve_player(player_id: object, small_id_by_player_id: dict[str, int]) -> int | None:
    if not isinstance(player_id, str):
        return None
    return small_id_by_player_id.get(player_id)


def _tile_xy(tile_ref: object, map_width: int) -> tuple[int | None, int | None]:
    """Engine TileRef -> ``(x, y)``, matching ``GameMap.x()``/``y()``
    (``ref = y * width + x``)."""
    if not isinstance(tile_ref, int) or isinstance(tile_ref, bool) or tile_ref < 0:
        return None, None
    return tile_ref % map_width, tile_ref // map_width


def encode_action(
    intent: dict[str, object] | None, *, small_id_by_player_id: dict[str, int], map_width: int
) -> EncodedAction | None:
    """Encodes one raw intent into an `EncodedAction`.

    Args:
        intent: a raw ``StampedIntent`` dict (see
            ``replay/snapshot_schema.py``), or ``None`` to encode the
            "no action taken this window" case.
        small_id_by_player_id: resolves a target field (``targetID``,
            ``recipient``, ``target``, ``requestor``) to the engine's
            smallID, which is what this project's tensors are keyed by
            everywhere else. Those fields hold engine **PlayerIDs**, not
            clientIDs — the two differ even for humans, and nations/tribes
            have no clientID at all. Build this from a
            ``TickSnapshot.players`` list:
            ``{p.id: p.smallID for p in tick.players}``.
        map_width: the game's ``SnapshotHeader.width``, needed to decode a
            TileRef into ``(target_x, target_y)``.

    Returns:
        ``None`` if the intent's type is in `UNMODELED_INTENT_TYPES` or
        unrecognized (the caller should skip this intent entirely, not
        treat it as noop); `NOOP_ACTION` if `intent` is `None`; otherwise
        the encoded action.
    """
    if intent is None:
        return NOOP_ACTION

    intent_type = intent.get("type")
    if intent_type in UNMODELED_INTENT_TYPES or intent_type not in _ACTION_TYPE_INDEX:
        return None
    assert isinstance(intent_type, str)  # narrowed by the `in _ACTION_TYPE_INDEX` check above

    unit_type = _UNIT_TYPE_INDEX["none"]
    target_player: int | None = None
    target_x: int | None = None
    target_y: int | None = None
    amount: float | None = None

    if intent_type in ("build_unit", "upgrade_structure"):
        unit_name = intent.get("unit")
        if isinstance(unit_name, str) and unit_name in _UNIT_TYPE_INDEX:
            unit_type = _UNIT_TYPE_INDEX[unit_name]

    if intent_type in TILE_TARGET_ACTION_TYPES:
        tile_field = "dst" if intent_type == "boat" else "tile"
        target_x, target_y = _tile_xy(intent.get(tile_field), map_width)

    if intent_type == "attack":
        target_id = intent.get("targetID")
        target_player = (
            TERRA_NULLIUS_SMALL_ID
            if target_id is None
            else _resolve_player(target_id, small_id_by_player_id)
        )
        amount = _as_float(intent.get("troops"))
    elif intent_type == "boat":
        amount = _as_float(intent.get("troops"))
    elif intent_type == "donate_gold":
        target_player = _resolve_player(intent.get("recipient"), small_id_by_player_id)
        amount = _as_float(intent.get("gold"))
    elif intent_type == "donate_troops":
        target_player = _resolve_player(intent.get("recipient"), small_id_by_player_id)
        amount = _as_float(intent.get("troops"))
    elif intent_type in ("allianceRequest", "allianceExtension", "breakAlliance"):
        target_player = _resolve_player(intent.get("recipient"), small_id_by_player_id)
    elif intent_type == "allianceReject":
        target_player = _resolve_player(intent.get("requestor"), small_id_by_player_id)
    elif intent_type == "embargo":
        target_player = _resolve_player(intent.get("targetID"), small_id_by_player_id)
    elif intent_type == "targetPlayer":
        target_player = _resolve_player(intent.get("target"), small_id_by_player_id)

    return EncodedAction(
        action_type=_ACTION_TYPE_INDEX[intent_type],
        unit_type=unit_type,
        target_x=target_x,
        target_y=target_y,
        target_player_small_id=target_player,
        amount=amount,
    )


def encode_legality_mask(
    legality: LegalityMask, *, spawn_phase: bool, viewer_num_tiles_owned: int
) -> np.ndarray:
    """Derives an `action_type`-level legality mask
    (``(len(ACTION_TYPES),)`` bool array) from a snapshot's coarse
    `LegalityMask`, plus the two spawn-eligibility facts `LegalityMask`
    itself doesn't carry.

    See ``docs/imitation_learning_plan.md`` Section 2 for exactly which
    action types have a real backing signal versus defaulting to legal —
    "legal" for an unbacked type means "uninformative," not "confirmed
    legal."
    """
    mask = np.ones(len(ACTION_TYPES), dtype=bool)

    def set_legal(name: str, value: bool) -> None:
        mask[_ACTION_TYPE_INDEX[name]] = value

    set_legal("spawn", spawn_phase and viewer_num_tiles_owned == 0)
    set_legal("attack", bool(legality.attackablePlayerSmallIDs) or legality.canExpand)
    set_legal("boat", bool(legality.attackablePlayerSmallIDs) or legality.canExpand)
    set_legal("build_unit", bool(legality.buildableUnitTypes))
    set_legal("upgrade_structure", bool(legality.buildableUnitTypes))
    set_legal("allianceRequest", bool(legality.allianceRequestablePlayerSmallIDs))
    set_legal("allianceReject", bool(legality.allianceAcceptablePlayerSmallIDs))
    set_legal("breakAlliance", bool(legality.breakableAlliancePlayerSmallIDs))
    set_legal("donate_gold", bool(legality.donatableGoldPlayerSmallIDs))
    set_legal("donate_troops", bool(legality.donatableTroopsPlayerSmallIDs))
    set_legal("embargo", bool(legality.embargoablePlayerSmallIDs))
    set_legal("embargo_all", legality.canEmbargoAll)
    set_legal("targetPlayer", bool(legality.targetablePlayerSmallIDs))
    set_legal("delete_unit", legality.canDeleteUnit)
    # noop, allianceExtension, move_warship, cancel_boat, cancel_attack:
    # no tracked signal in LegalityMask, left at the default (legal).
    return mask


def encode_tile_target_mask(spatial: np.ndarray, action_type: int, unit_type: int) -> np.ndarray:
    """``(H, W)`` bool mask of tiles that are plausible targets for
    ``action_type`` (with ``unit_type`` for ``build_unit``), derived from
    an ``encode_spatial()`` tensor. All ``False`` for an action type not in
    ``TILE_TARGET_ACTION_TYPES``.

    Deliberately coarse — a necessary condition, not the engine's full
    rule (no build-spacing, port-adjacency or range checks): spawn on
    unowned land, build on your own territory (nukes: anywhere; Warship:
    water), boat to land you don't own, warships to water.
    Training still never masks out the true label (see
    ``training/bc_trainer.py``'s ``masked_cross_entropy``), so a rule
    that's too strict costs probability mass, never a ``NaN``.
    """
    channel: dict[str, np.ndarray] = {
        name: np.asarray(spatial[index] > 0) for index, name in enumerate(SPATIAL_CHANNELS)
    }
    is_land = channel["is_land"]
    is_owned: np.ndarray = (
        channel["owned_by_viewer"] | channel["owned_by_ally"] | channel["owned_by_enemy"]
    )
    not_land: np.ndarray = ~is_land
    name = ACTION_TYPES[action_type]

    if name == "spawn":
        return np.asarray(is_land & ~is_owned)
    if name == "boat":
        return np.asarray(is_land & ~channel["owned_by_viewer"])
    if name == "move_warship":
        return not_land
    if name == "build_unit":
        unit_name = UNIT_TYPES[unit_type]
        if unit_name in _NUKE_UNIT_TYPES:
            return np.ones_like(is_land)
        if unit_name == "Warship":
            return not_land
        return channel["owned_by_viewer"]
    return np.zeros_like(is_land)


def encode_player_target_mask(
    legality: LegalityMask, candidate_small_ids: np.ndarray, action_type: int
) -> np.ndarray:
    """``(N,)`` bool mask over ``observation_encoder.encode_player_candidates``'
    candidates: which are legal targets for ``action_type``, per
    ``legality``'s per-action player lists. All ``False`` for an action
    type not in ``PLAYER_TARGET_ACTION_TYPES``.

    Terra nullius is only ever a legal target for ``attack`` (and only
    when ``legality.canExpand``). ``allianceExtension`` has no tracked
    signal, so every real player defaults to legal for it — "legal" there
    means "uninformative," as in ``encode_legality_mask``.
    """
    name = ACTION_TYPES[action_type]
    is_player: np.ndarray = np.asarray(candidate_small_ids != TERRA_NULLIUS_SMALL_ID)
    legal_lists = {
        "attack": legality.attackablePlayerSmallIDs,
        "allianceRequest": legality.allianceRequestablePlayerSmallIDs,
        "allianceReject": legality.allianceAcceptablePlayerSmallIDs,
        "breakAlliance": legality.breakableAlliancePlayerSmallIDs,
        "donate_gold": legality.donatableGoldPlayerSmallIDs,
        "donate_troops": legality.donatableTroopsPlayerSmallIDs,
        "embargo": legality.embargoablePlayerSmallIDs,
        "targetPlayer": legality.targetablePlayerSmallIDs,
    }

    if name == "allianceExtension":
        return is_player
    if name not in legal_lists:
        return np.zeros(len(candidate_small_ids), dtype=bool)
    mask: np.ndarray = np.isin(candidate_small_ids, legal_lists[name]) & is_player
    if name == "attack" and legality.canExpand:
        mask |= ~is_player
    return mask
