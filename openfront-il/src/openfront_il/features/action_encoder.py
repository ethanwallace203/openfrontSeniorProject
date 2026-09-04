"""Pure functions: engine action -> multi-head discrete action label(s) +
legality mask.

See ``docs/imitation_learning_plan.md`` Section 2 for the full taxonomy
design and the reasoning behind every scope decision here — most notably
why tile/player targets are passed through raw rather than one-hot
encoded (a fixed-size classification head over tiles or over a
variable-size, per-game player roster isn't tractable at this scale).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


def _resolve_player(client_id: object, small_id_by_client_id: dict[str, int]) -> int | None:
    if not isinstance(client_id, str):
        return None
    return small_id_by_client_id.get(client_id)


def encode_action(
    intent: dict[str, object] | None, *, small_id_by_client_id: dict[str, int]
) -> EncodedAction | None:
    """Encodes one raw intent into an `EncodedAction`.

    Args:
        intent: a raw ``StampedIntent`` dict (see
            ``replay/snapshot_schema.py``), or ``None`` to encode the
            "no action taken this window" case.
        small_id_by_client_id: resolves a clientID-valued target field
            (e.g. ``recipient``, ``targetID``) to the engine's smallID,
            which is what this project's tensors are keyed by everywhere
            else. Build this from a ``TickSnapshot.players`` list:
            ``{p.clientID: p.smallID for p in tick.players if p.clientID}``.

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
    amount: float | None = None

    if intent_type in ("build_unit", "upgrade_structure"):
        unit_name = intent.get("unit")
        if isinstance(unit_name, str) and unit_name in _UNIT_TYPE_INDEX:
            unit_type = _UNIT_TYPE_INDEX[unit_name]

    if intent_type == "attack":
        target_player = _resolve_player(intent.get("targetID"), small_id_by_client_id)
        amount = _as_float(intent.get("troops"))
    elif intent_type == "boat":
        amount = _as_float(intent.get("troops"))
    elif intent_type == "donate_gold":
        target_player = _resolve_player(intent.get("recipient"), small_id_by_client_id)
        amount = _as_float(intent.get("gold"))
    elif intent_type == "donate_troops":
        target_player = _resolve_player(intent.get("recipient"), small_id_by_client_id)
        amount = _as_float(intent.get("troops"))
    elif intent_type in ("allianceRequest", "allianceExtension", "breakAlliance"):
        target_player = _resolve_player(intent.get("recipient"), small_id_by_client_id)
    elif intent_type == "allianceReject":
        target_player = _resolve_player(intent.get("requestor"), small_id_by_client_id)
    elif intent_type == "embargo":
        target_player = _resolve_player(intent.get("targetID"), small_id_by_client_id)
    elif intent_type == "targetPlayer":
        target_player = _resolve_player(intent.get("target"), small_id_by_client_id)

    return EncodedAction(
        action_type=_ACTION_TYPE_INDEX[intent_type],
        unit_type=unit_type,
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
