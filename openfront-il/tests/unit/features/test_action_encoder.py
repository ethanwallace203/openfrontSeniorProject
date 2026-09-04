import numpy as np

from openfront_il.features.action_encoder import (
    ACTION_TYPES,
    NOOP_ACTION,
    UNIT_TYPES,
    encode_action,
    encode_legality_mask,
)
from openfront_il.replay.snapshot_schema import LegalityMask

CLIENTS = {"alice": 1, "bob": 2, "carol": 3}


def _legality(**overrides: object) -> LegalityMask:
    defaults: dict[str, object] = dict(
        attackablePlayerSmallIDs=[],
        canExpand=False,
        allianceRequestablePlayerSmallIDs=[],
        allianceAcceptablePlayerSmallIDs=[],
        breakableAlliancePlayerSmallIDs=[],
        donatableGoldPlayerSmallIDs=[],
        donatableTroopsPlayerSmallIDs=[],
        embargoablePlayerSmallIDs=[],
        targetablePlayerSmallIDs=[],
        buildableUnitTypes=[],
        canDeleteUnit=False,
        canEmbargoAll=False,
    )
    defaults.update(overrides)
    return LegalityMask.model_validate(defaults)


class TestEncodeAction:
    def test_none_intent_is_noop(self) -> None:
        assert encode_action(None, small_id_by_client_id=CLIENTS) == NOOP_ACTION

    def test_unmodeled_intent_returns_none(self) -> None:
        assert (
            encode_action({"type": "emoji", "clientID": "alice"}, small_id_by_client_id=CLIENTS)
            is None
        )
        assert (
            encode_action(
                {"type": "quick_chat", "clientID": "alice"}, small_id_by_client_id=CLIENTS
            )
            is None
        )

    def test_unrecognized_intent_returns_none(self) -> None:
        assert (
            encode_action({"type": "some_future_intent_type"}, small_id_by_client_id=CLIENTS)
            is None
        )

    def test_attack_resolves_target_and_troops(self) -> None:
        intent = {"type": "attack", "targetID": "bob", "troops": 500.0, "clientID": "alice"}

        action = encode_action(intent, small_id_by_client_id=CLIENTS)

        assert action is not None
        assert action.action_type == ACTION_TYPES.index("attack")
        assert action.target_player_small_id == 2
        assert action.amount == 500.0

    def test_attack_on_terra_nullius_has_no_target_player(self) -> None:
        intent = {"type": "attack", "targetID": None, "troops": 100.0, "clientID": "alice"}

        action = encode_action(intent, small_id_by_client_id=CLIENTS)

        assert action is not None
        assert action.target_player_small_id is None

    def test_build_unit_resolves_unit_type(self) -> None:
        intent = {"type": "build_unit", "unit": "Missile Silo", "tile": 12345, "clientID": "alice"}

        action = encode_action(intent, small_id_by_client_id=CLIENTS)

        assert action is not None
        assert action.action_type == ACTION_TYPES.index("build_unit")
        assert action.unit_type == UNIT_TYPES.index("Missile Silo")

    def test_build_unit_with_unknown_unit_name_falls_back_to_none(self) -> None:
        intent = {"type": "build_unit", "unit": "Some Future Unit", "clientID": "alice"}

        action = encode_action(intent, small_id_by_client_id=CLIENTS)

        assert action is not None
        assert action.unit_type == UNIT_TYPES.index("none")

    def test_donate_gold_resolves_recipient_and_amount(self) -> None:
        intent = {"type": "donate_gold", "recipient": "carol", "gold": 250, "clientID": "alice"}

        action = encode_action(intent, small_id_by_client_id=CLIENTS)

        assert action is not None
        assert action.target_player_small_id == 3
        assert action.amount == 250.0

    def test_alliance_reject_resolves_requestor_not_recipient(self) -> None:
        intent = {"type": "allianceReject", "requestor": "bob", "clientID": "alice"}

        action = encode_action(intent, small_id_by_client_id=CLIENTS)

        assert action is not None
        assert action.target_player_small_id == 2

    def test_unresolvable_client_id_gives_none_target(self) -> None:
        intent = {
            "type": "attack",
            "targetID": "someone-not-in-this-game",
            "troops": 1,
            "clientID": "alice",
        }

        action = encode_action(intent, small_id_by_client_id=CLIENTS)

        assert action is not None
        assert action.target_player_small_id is None


class TestEncodeLegalityMask:
    def test_defaults_to_legal_for_unbacked_types(self) -> None:
        mask = encode_legality_mask(_legality(), spawn_phase=False, viewer_num_tiles_owned=1)

        for name in ("noop", "allianceExtension", "move_warship", "cancel_boat", "cancel_attack"):
            assert mask[ACTION_TYPES.index(name)]

    def test_spawn_legal_only_during_spawn_phase_with_no_tiles(self) -> None:
        mask = encode_legality_mask(_legality(), spawn_phase=True, viewer_num_tiles_owned=0)
        assert mask[ACTION_TYPES.index("spawn")]

        mask = encode_legality_mask(_legality(), spawn_phase=True, viewer_num_tiles_owned=5)
        assert not mask[ACTION_TYPES.index("spawn")]

        mask = encode_legality_mask(_legality(), spawn_phase=False, viewer_num_tiles_owned=0)
        assert not mask[ACTION_TYPES.index("spawn")]

    def test_attack_and_boat_track_attackable_players_and_expand(self) -> None:
        mask = encode_legality_mask(
            _legality(attackablePlayerSmallIDs=[2]), spawn_phase=False, viewer_num_tiles_owned=5
        )
        assert mask[ACTION_TYPES.index("attack")]
        assert mask[ACTION_TYPES.index("boat")]

        mask = encode_legality_mask(
            _legality(canExpand=True), spawn_phase=False, viewer_num_tiles_owned=5
        )
        assert mask[ACTION_TYPES.index("attack")]

        mask = encode_legality_mask(_legality(), spawn_phase=False, viewer_num_tiles_owned=5)
        assert not mask[ACTION_TYPES.index("attack")]
        assert not mask[ACTION_TYPES.index("boat")]

    def test_build_and_upgrade_track_buildable_unit_types(self) -> None:
        mask = encode_legality_mask(
            _legality(buildableUnitTypes=["City"]), spawn_phase=False, viewer_num_tiles_owned=5
        )
        assert mask[ACTION_TYPES.index("build_unit")]
        assert mask[ACTION_TYPES.index("upgrade_structure")]

    def test_returns_correct_length_boolean_array(self) -> None:
        mask = encode_legality_mask(_legality(), spawn_phase=False, viewer_num_tiles_owned=5)
        assert mask.shape == (len(ACTION_TYPES),)
        assert mask.dtype == np.bool_
