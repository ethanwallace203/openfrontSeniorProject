import numpy as np
import pytest

from openfront_il.features.action_encoder import (
    ACTION_TYPES,
    AMOUNT_BUCKET_EDGES,
    AMOUNT_BUCKET_VALUES,
    NOOP_ACTION,
    PLAYER_TARGET_ACTION_TYPES,
    PORT_SHORE_RADIUS,
    TILE_TARGET_ACTION_TYPES,
    UNIT_TYPES,
    decode_amount_bucket,
    encode_action,
    encode_amount_bucket,
    encode_legality_mask,
    encode_player_target_mask,
    encode_tile_target_mask,
)
from openfront_il.features.observation_encoder import SPATIAL_CHANNELS, TERRA_NULLIUS_SMALL_ID
from openfront_il.replay.snapshot_schema import LegalityMask

# Intent target fields hold engine PlayerIDs, which differ from clientIDs
# (e.g. the issuing human "alice" below has PlayerID "pid-alice").
PLAYERS = {"pid-alice": 1, "pid-bob": 2, "pid-carol": 3}
WIDTH = 100


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
        assert encode_action(None, small_id_by_player_id=PLAYERS, map_width=WIDTH) == NOOP_ACTION

    def test_unmodeled_intent_returns_none(self) -> None:
        assert (
            encode_action(
                {"type": "emoji", "clientID": "alice"},
                small_id_by_player_id=PLAYERS,
                map_width=WIDTH,
            )
            is None
        )
        assert (
            encode_action(
                {"type": "quick_chat", "clientID": "alice"},
                small_id_by_player_id=PLAYERS,
                map_width=WIDTH,
            )
            is None
        )

    def test_unrecognized_intent_returns_none(self) -> None:
        assert (
            encode_action(
                {"type": "some_future_intent_type"}, small_id_by_player_id=PLAYERS, map_width=WIDTH
            )
            is None
        )

    def test_attack_resolves_target_and_troops(self) -> None:
        intent = {"type": "attack", "targetID": "pid-bob", "troops": 500.0, "clientID": "alice"}

        action = encode_action(intent, small_id_by_player_id=PLAYERS, map_width=WIDTH)

        assert action is not None
        assert action.action_type == ACTION_TYPES.index("attack")
        assert action.target_player_small_id == 2
        assert action.amount == 500.0

    def test_attack_on_unowned_land_targets_terra_nullius(self) -> None:
        """A null targetID is "expand into unowned land" — the most common
        attack in the game, so it's a real target_player label, not a
        missing one."""
        intent = {"type": "attack", "targetID": None, "troops": 100.0, "clientID": "alice"}

        action = encode_action(intent, small_id_by_player_id=PLAYERS, map_width=WIDTH)

        assert action is not None
        assert action.target_player_small_id == TERRA_NULLIUS_SMALL_ID

    def test_build_unit_resolves_unit_type(self) -> None:
        intent = {"type": "build_unit", "unit": "Missile Silo", "tile": 12345, "clientID": "alice"}

        action = encode_action(intent, small_id_by_player_id=PLAYERS, map_width=WIDTH)

        assert action is not None
        assert action.action_type == ACTION_TYPES.index("build_unit")
        assert action.unit_type == UNIT_TYPES.index("Missile Silo")

    def test_build_unit_with_unknown_unit_name_falls_back_to_none(self) -> None:
        intent = {"type": "build_unit", "unit": "Some Future Unit", "clientID": "alice"}

        action = encode_action(intent, small_id_by_player_id=PLAYERS, map_width=WIDTH)

        assert action is not None
        assert action.unit_type == UNIT_TYPES.index("none")

    def test_donate_gold_resolves_recipient_and_amount(self) -> None:
        intent = {"type": "donate_gold", "recipient": "pid-carol", "gold": 250, "clientID": "alice"}

        action = encode_action(intent, small_id_by_player_id=PLAYERS, map_width=WIDTH)

        assert action is not None
        assert action.target_player_small_id == 3
        assert action.amount == 250.0

    def test_alliance_reject_resolves_requestor_not_recipient(self) -> None:
        intent = {"type": "allianceReject", "requestor": "pid-bob", "clientID": "alice"}

        action = encode_action(intent, small_id_by_player_id=PLAYERS, map_width=WIDTH)

        assert action is not None
        assert action.target_player_small_id == 2

    def test_client_id_is_not_a_target_identifier(self) -> None:
        """Regression: targets used to be looked up by clientID, which never
        matches a real intent's targetID (a PlayerID) — every player target
        silently resolved to None."""
        intent = {"type": "attack", "targetID": "bob", "troops": 1.0, "clientID": "alice"}

        action = encode_action(intent, small_id_by_player_id=PLAYERS, map_width=WIDTH)

        assert action is not None
        assert action.target_player_small_id is None

    def test_unresolvable_player_id_gives_none_target(self) -> None:
        intent = {
            "type": "attack",
            "targetID": "someone-not-in-this-game",
            "troops": 1,
            "clientID": "alice",
        }

        action = encode_action(intent, small_id_by_player_id=PLAYERS, map_width=WIDTH)

        assert action is not None
        assert action.target_player_small_id is None


class TestEncodeActionTileTargets:
    @pytest.mark.parametrize(
        ("intent", "expected_xy"),
        [
            ({"type": "spawn", "tile": 2 * WIDTH + 7}, (7, 2)),
            ({"type": "build_unit", "unit": "City", "tile": 5 * WIDTH}, (0, 5)),
            ({"type": "move_warship", "unitIds": [1], "tile": 99}, (99, 0)),
            ({"type": "boat", "troops": 10.0, "dst": 3 * WIDTH + 4}, (4, 3)),
        ],
    )
    def test_decodes_tile_ref_to_xy(
        self, intent: dict[str, object], expected_xy: tuple[int, int]
    ) -> None:
        action = encode_action(intent, small_id_by_player_id=PLAYERS, map_width=WIDTH)

        assert action is not None
        assert (action.target_x, action.target_y) == expected_xy

    def test_missing_tile_gives_no_target(self) -> None:
        action = encode_action(
            {"type": "build_unit", "unit": "City"}, small_id_by_player_id=PLAYERS, map_width=WIDTH
        )

        assert action is not None
        assert action.target_x is None
        assert action.target_y is None

    def test_non_tile_action_gives_no_target(self) -> None:
        action = encode_action(
            {"type": "attack", "targetID": "pid-bob", "troops": 1.0},
            small_id_by_player_id=PLAYERS,
            map_width=WIDTH,
        )

        assert action is not None
        assert action.target_x is None


class TestArgumentActionTypeSets:
    def test_are_known_action_types_and_disjoint(self) -> None:
        assert TILE_TARGET_ACTION_TYPES <= set(ACTION_TYPES)
        assert PLAYER_TARGET_ACTION_TYPES <= set(ACTION_TYPES)
        assert TILE_TARGET_ACTION_TYPES.isdisjoint(PLAYER_TARGET_ACTION_TYPES)
        assert "noop" not in TILE_TARGET_ACTION_TYPES | PLAYER_TARGET_ACTION_TYPES


def _spatial(**channels: list[list[float]]) -> np.ndarray:
    """A 2x2 spatial tensor with the named channels set, others zero."""
    spatial = np.zeros((len(SPATIAL_CHANNELS), 2, 2), dtype=np.float32)
    for name, values in channels.items():
        spatial[SPATIAL_CHANNELS.index(name)] = values
    return spatial


class TestEncodeTileTargetMask:
    # Top row land (left: mine, right: enemy's); bottom row water.
    SPATIAL = _spatial(
        is_land=[[1, 1], [0, 0]],
        owned_by_viewer=[[1, 0], [0, 0]],
        owned_by_enemy=[[0, 1], [0, 0]],
    )

    def _mask(self, action: str, unit: str = "none") -> list[list[bool]]:
        mask = encode_tile_target_mask(
            self.SPATIAL, ACTION_TYPES.index(action), UNIT_TYPES.index(unit)
        )
        assert mask.shape == (2, 2)
        assert mask.dtype == np.bool_
        return mask.tolist()

    def test_build_structure_on_own_territory(self) -> None:
        assert self._mask("build_unit", "City") == [[True, False], [False, False]]

    def test_build_nuke_anywhere_but_own_territory(self) -> None:
        assert self._mask("build_unit", "Atom Bomb") == [[False, True], [True, True]]

    def test_build_warship_and_move_warship_on_water(self) -> None:
        water = [[False, False], [True, True]]
        assert self._mask("build_unit", "Warship") == water
        assert self._mask("move_warship") == water

    def test_boat_to_land_not_owned_by_viewer(self) -> None:
        assert self._mask("boat") == [[False, True], [False, False]]

    def test_boat_never_targets_an_ally(self) -> None:
        spatial = _spatial(is_land=[[1, 1], [1, 0]], owned_by_ally=[[0, 1], [0, 0]])
        mask = encode_tile_target_mask(spatial, ACTION_TYPES.index("boat"), 0)
        # Inland unowned land is fine (the engine finds the shore itself).
        assert mask.tolist() == [[True, False], [True, False]]

    def test_port_only_within_reach_of_own_coast(self) -> None:
        """A 1-row strip of own land, shoreline at x=0 only: tiles within
        PORT_SHORE_RADIUS of it qualify, tiles beyond don't."""
        width = PORT_SHORE_RADIUS + 5
        spatial = np.zeros((len(SPATIAL_CHANNELS), 1, width), dtype=np.float32)
        spatial[SPATIAL_CHANNELS.index("is_land")] = 1
        spatial[SPATIAL_CHANNELS.index("owned_by_viewer")] = 1
        spatial[SPATIAL_CHANNELS.index("is_shoreline"), 0, 0] = 1

        mask = encode_tile_target_mask(
            spatial, ACTION_TYPES.index("build_unit"), UNIT_TYPES.index("Port")
        )[0]

        assert mask[: PORT_SHORE_RADIUS + 1].all()
        assert not mask[PORT_SHORE_RADIUS + 1 :].any()

    def test_port_needs_your_own_coast(self) -> None:
        # Own land, but the only shoreline is an enemy's.
        spatial = _spatial(
            is_land=[[1, 1], [0, 0]],
            owned_by_viewer=[[1, 0], [0, 0]],
            owned_by_enemy=[[0, 1], [0, 0]],
            is_shoreline=[[0, 1], [0, 0]],
        )
        mask = encode_tile_target_mask(
            spatial, ACTION_TYPES.index("build_unit"), UNIT_TYPES.index("Port")
        )
        assert not mask.any()

    def test_spawn_on_unowned_land_only(self) -> None:
        spatial = _spatial(is_land=[[1, 1], [0, 0]], owned_by_enemy=[[0, 1], [0, 0]])
        mask = encode_tile_target_mask(spatial, ACTION_TYPES.index("spawn"), 0)
        assert mask.tolist() == [[True, False], [False, False]]

    def test_non_tile_action_is_all_false(self) -> None:
        assert self._mask("attack") == [[False, False], [False, False]]


class TestEncodePlayerTargetMask:
    CANDIDATES = np.array([TERRA_NULLIUS_SMALL_ID, 2, 3], dtype=np.int64)

    def _mask(self, action: str, **legality: object) -> list[bool]:
        mask = encode_player_target_mask(
            _legality(**legality), self.CANDIDATES, ACTION_TYPES.index(action)
        )
        assert mask.dtype == np.bool_
        return mask.tolist()

    def test_attack_uses_attackable_players(self) -> None:
        assert self._mask("attack", attackablePlayerSmallIDs=[3]) == [False, False, True]

    def test_attack_allows_terra_nullius_only_when_can_expand(self) -> None:
        assert self._mask("attack", canExpand=True) == [True, False, False]

    def test_terra_nullius_never_legal_for_non_attack(self) -> None:
        assert self._mask("donate_gold", donatableGoldPlayerSmallIDs=[0, 2]) == [
            False,
            True,
            False,
        ]

    def test_alliance_reject_uses_acceptable_list(self) -> None:
        assert self._mask("allianceReject", allianceAcceptablePlayerSmallIDs=[2]) == [
            False,
            True,
            False,
        ]

    def test_alliance_extension_defaults_every_player_legal(self) -> None:
        assert self._mask("allianceExtension") == [False, True, True]

    def test_non_player_action_is_all_false(self) -> None:
        assert self._mask("build_unit", attackablePlayerSmallIDs=[2, 3]) == [False, False, False]


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


class TestAmountBuckets:
    ATTACK = ACTION_TYPES.index("attack")
    DONATE_GOLD = ACTION_TYPES.index("donate_gold")
    NOOP = ACTION_TYPES.index("noop")

    def _bucket(self, action_type: int, amount: float | None, troops: float = 1000.0) -> int:
        return encode_amount_bucket(action_type, amount, viewer_troops=troops, viewer_gold=500.0)

    def test_fraction_of_troops_picks_the_bucket(self) -> None:
        # 245 of 1000 troops = 0.245 -> the 0.23-0.26 bucket, the human peak.
        assert self._bucket(self.ATTACK, 245.0) == AMOUNT_BUCKET_EDGES.index(0.23) + 1

    def test_bucket_edges_are_lower_inclusive(self) -> None:
        assert self._bucket(self.ATTACK, 49.0) == 0
        assert self._bucket(self.ATTACK, 50.0) == 1

    def test_more_than_everything_lands_in_the_top_bucket(self) -> None:
        # Troops change inside the snapshot window, so >100% happens.
        top = len(AMOUNT_BUCKET_VALUES) - 1
        assert self._bucket(self.ATTACK, 1000.0) == top
        assert self._bucket(self.ATTACK, 5000.0) == top

    def test_donate_gold_is_a_fraction_of_gold(self) -> None:
        # 250 of 500 gold = 0.5 -> the 0.50-0.75 bucket.
        assert self._bucket(self.DONATE_GOLD, 250.0) == AMOUNT_BUCKET_EDGES.index(0.50) + 1

    def test_no_label_cases(self) -> None:
        assert self._bucket(self.NOOP, 100.0) == -1  # takes no amount
        assert self._bucket(self.ATTACK, None) == -1  # intent didn't carry one
        assert self._bucket(self.ATTACK, 100.0, troops=0.0) == -1  # nothing to divide by

    def test_decode_returns_the_bucket_value_times_what_the_viewer_has(self) -> None:
        bucket = self._bucket(self.ATTACK, 245.0)
        troops = decode_amount_bucket(self.ATTACK, bucket, viewer_troops=2000.0, viewer_gold=0.0)
        assert troops == pytest.approx(AMOUNT_BUCKET_VALUES[bucket] * 2000.0)

    def test_every_bucket_value_round_trips_to_its_own_bucket(self) -> None:
        for bucket, value in enumerate(AMOUNT_BUCKET_VALUES):
            assert self._bucket(self.ATTACK, value * 1000.0) == bucket
