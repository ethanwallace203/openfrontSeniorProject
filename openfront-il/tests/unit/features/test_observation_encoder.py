import base64
import gzip

import numpy as np
import pytest

from openfront_il.features.observation_encoder import (
    FLAT_FEATURES,
    PLAYER_FEATURES,
    SPATIAL_CHANNELS,
    TERRA_NULLIUS_SMALL_ID,
    encode_candidate_cells,
    encode_flat,
    encode_observation,
    encode_player_candidates,
    encode_spatial,
)
from openfront_il.replay.snapshot_schema import (
    AllianceSnapshot,
    AttackSnapshot,
    PlayerScalars,
    SnapshotHeader,
    TickSnapshot,
    UnitSnapshot,
)

# Bit positions, matching engine/openfront-ai/openfront/src/core/game/GameMap.ts
_IS_LAND_BIT = 7
_SHORELINE_BIT = 6
_FALLOUT_BIT = 13
_DEFENSE_BONUS_BIT = 14


def _gzip_b64(data: bytes) -> str:
    return base64.b64encode(gzip.compress(data)).decode("ascii")


def _make_header(terrain_bytes: list[int], width: int, height: int) -> SnapshotHeader:
    return SnapshotHeader(
        kind="header",
        formatVersion=2,
        gameId="test-game",
        gitCommit="deadbeef",
        map="TestMap",
        width=width,
        height=height,
        terrainGzipBase64=_gzip_b64(bytes(terrain_bytes)),
        snapshotEvery=10,
    )


def _make_tick(
    tile_states: list[int],
    *,
    players: list[PlayerScalars],
    alliances: list[AllianceSnapshot] | None = None,
    units: list[UnitSnapshot] | None = None,
    spawn_phase: bool = False,
) -> TickSnapshot:
    raw = np.array(tile_states, dtype="<u2").tobytes()
    return TickSnapshot(
        kind="tick",
        tick=10,
        spawnPhase=spawn_phase,
        tileStateGzipBase64=_gzip_b64(raw),
        players=players,
        units=units or [],
        attacks=[],
        alliances=alliances or [],
        actions={},
        legal={},
        winner=None,
    )


def _player(small_id: int, **overrides: object) -> PlayerScalars:
    defaults: dict[str, object] = dict(
        id=f"pid{small_id}",
        clientID=f"client{small_id}",
        smallID=small_id,
        username=f"player{small_id}",
        type="HUMAN",
        team=None,
        isAlive=True,
        isTraitor=False,
        isDisconnected=False,
        numTilesOwned=0,
        gold="0",
        troops=0.0,
    )
    defaults.update(overrides)
    return PlayerScalars.model_validate(defaults)


class TestEncodeSpatial:
    def test_is_uint8(self) -> None:
        header = _make_header([0, 0], width=2, height=1)
        tick = _make_tick([1, 0], players=[_player(1)])

        assert encode_spatial(header, tick, viewer_small_id=1).dtype == np.uint8

    def test_terrain_and_ownership_channels(self) -> None:
        # A 2x2 map: tile 0 is land+shoreline, tile 1 is land only,
        # tile 2 is not land, tile 3 is not land.
        terrain = [
            (1 << _IS_LAND_BIT) | (1 << _SHORELINE_BIT),
            1 << _IS_LAND_BIT,
            0,
            0,
        ]
        header = _make_header(terrain, width=2, height=2)

        # Owner IDs: tile 0 -> viewer(1), tile 1 -> ally(2), tile 2 -> enemy(3),
        # tile 3 -> unowned (0). Tile 0 also has fallout+defense bonus set.
        tile_states = [
            1 | (1 << _FALLOUT_BIT) | (1 << _DEFENSE_BONUS_BIT),
            2,
            3,
            0,
        ]
        players = [_player(1), _player(2), _player(3)]
        alliances = [AllianceSnapshot(requestorSmallID=1, recipientSmallID=2, expiresAtTick=1000)]
        tick = _make_tick(tile_states, players=players, alliances=alliances)

        spatial = encode_spatial(header, tick, viewer_small_id=1)

        assert spatial.shape == (len(SPATIAL_CHANNELS), 2, 2)

        def channel(name: str) -> np.ndarray:
            return spatial[SPATIAL_CHANNELS.index(name)]

        np.testing.assert_array_equal(channel("is_land").flatten(), [1, 1, 0, 0])
        np.testing.assert_array_equal(channel("is_shoreline").flatten(), [1, 0, 0, 0])
        np.testing.assert_array_equal(channel("owned_by_viewer").flatten(), [1, 0, 0, 0])
        np.testing.assert_array_equal(channel("owned_by_ally").flatten(), [0, 1, 0, 0])
        np.testing.assert_array_equal(channel("owned_by_enemy").flatten(), [0, 0, 1, 0])
        np.testing.assert_array_equal(channel("has_fallout").flatten(), [1, 0, 0, 0])
        np.testing.assert_array_equal(channel("has_defense_bonus").flatten(), [1, 0, 0, 0])

    def test_no_allies_gives_all_zero_ally_channel(self) -> None:
        header = _make_header([0, 0], width=2, height=1)
        tick = _make_tick([1, 3], players=[_player(1), _player(3)])

        spatial = encode_spatial(header, tick, viewer_small_id=1)

        ally_channel = spatial[SPATIAL_CHANNELS.index("owned_by_ally")]
        assert not ally_channel.any()


class TestEncodeFlat:
    def test_basic_scalars(self) -> None:
        viewer = _player(1, gold="1000", troops=50.0, numTilesOwned=10, isAlive=True)
        enemy = _player(2, troops=30.0, numTilesOwned=5, isAlive=True)
        dead_enemy = _player(3, troops=999.0, numTilesOwned=999, isAlive=False)
        ally = _player(4, isAlive=True)
        tick = _make_tick(
            [1, 2, 3, 4],
            players=[viewer, enemy, dead_enemy, ally],
            alliances=[AllianceSnapshot(requestorSmallID=1, recipientSmallID=4, expiresAtTick=100)],
        )

        flat = encode_flat(tick, viewer_small_id=1)

        assert flat.shape == (len(FLAT_FEATURES),)
        values = dict(zip(FLAT_FEATURES, flat, strict=True))
        assert values["viewer_gold_log1p"] == pytest.approx(np.log1p(1000.0))
        assert values["viewer_troops_log1p"] == pytest.approx(np.log1p(50.0))
        assert values["viewer_num_tiles_owned_log1p"] == pytest.approx(np.log1p(10.0))
        assert values["viewer_is_alive"] == 1.0
        assert values["viewer_num_alliances"] == 1.0
        # dead_enemy must not count toward enemy totals.
        assert values["enemy_total_tiles_owned_log1p"] == pytest.approx(np.log1p(5.0))
        assert values["enemy_total_troops_log1p"] == pytest.approx(np.log1p(30.0))
        assert values["enemy_alive_count"] == 1.0
        assert values["ally_alive_count"] == 1.0

    def test_units_in_flight_counts_only_viewers_units_with_a_target(self) -> None:
        viewer = _player(1)
        units = [
            UnitSnapshot(
                id=1,
                type="Transport",
                ownerSmallID=1,
                x=0,
                y=0,
                targetX=5,
                targetY=5,
                health=100,
                maxHealth=100,
                troops=10,
                level=1,
                isUnderConstruction=False,
            ),
            UnitSnapshot(
                id=2,
                type="City",
                ownerSmallID=1,
                x=1,
                y=1,
                targetX=None,
                targetY=None,
                health=100,
                maxHealth=100,
                troops=0,
                level=1,
                isUnderConstruction=False,
            ),
            UnitSnapshot(
                id=3,
                type="Transport",
                ownerSmallID=2,
                x=2,
                y=2,
                targetX=3,
                targetY=3,
                health=100,
                maxHealth=100,
                troops=5,
                level=1,
                isUnderConstruction=False,
            ),
        ]
        tick = _make_tick([1], players=[viewer], units=units)

        flat = encode_flat(tick, viewer_small_id=1)

        assert flat[FLAT_FEATURES.index("viewer_units_in_flight")] == 1.0

    def test_raises_for_unknown_viewer(self) -> None:
        tick = _make_tick([1], players=[_player(1)])
        try:
            encode_flat(tick, viewer_small_id=99)
        except ValueError:
            return
        raise AssertionError("expected ValueError")


LAND = 1 << _IS_LAND_BIT


class TestEncodePlayerCandidates:
    """A 1x4 all-land strip owned [unowned, viewer(1), enemy(2), ally(3)]
    plus a dead player 4."""

    def _encode(self, **tick_overrides: object) -> tuple[np.ndarray, np.ndarray]:
        header = _make_header([LAND] * 4, width=4, height=1)
        tick = _make_tick(
            [0, 1, 2, 3],
            players=[
                _player(3),
                _player(1, troops=1000.0),
                _player(2, troops=10.0, numTilesOwned=1, clientID=None),
                _player(4, isAlive=False),
            ],
            alliances=[AllianceSnapshot(requestorSmallID=1, recipientSmallID=3, expiresAtTick=999)],
        )
        tick = tick.model_copy(update=tick_overrides)
        return encode_player_candidates(header, tick, viewer_small_id=1)

    def _feature(self, features: np.ndarray, row: int, name: str) -> float:
        return float(features[row, PLAYER_FEATURES.index(name)])

    def test_terra_nullius_first_then_living_others_by_small_id(self) -> None:
        small_ids, features = self._encode()

        assert small_ids.tolist() == [TERRA_NULLIUS_SMALL_ID, 2, 3]
        assert features.shape == (3, len(PLAYER_FEATURES))
        assert features.dtype == np.float32

    def test_relation_flags(self) -> None:
        _, features = self._encode()

        assert self._feature(features, 0, "is_terra_nullius") == 1.0
        assert self._feature(features, 1, "is_ally") == 0.0
        assert self._feature(features, 2, "is_ally") == 1.0
        assert self._feature(features, 1, "is_human") == 0.0  # no clientID: a nation/bot
        assert self._feature(features, 2, "is_human") == 1.0

    def test_shared_border_counts_land_edges_with_viewer(self) -> None:
        _, features = self._encode()

        # The viewer's single tile touches unowned land and player 2, not player 3.
        assert self._feature(features, 0, "shared_border_log1p") == pytest.approx(np.log1p(1))
        assert self._feature(features, 1, "shared_border_log1p") == pytest.approx(np.log1p(1))
        assert self._feature(features, 2, "shared_border_log1p") == 0.0

    def test_troop_advantage_is_relative_to_viewer(self) -> None:
        _, features = self._encode()

        expected = np.log1p(1000.0) - np.log1p(10.0)
        assert self._feature(features, 1, "troop_advantage_log_ratio") == pytest.approx(expected)

    def test_attack_flags(self) -> None:
        attacks = [
            AttackSnapshot(
                id="a",
                attackerSmallID=2,
                targetSmallID=1,
                troops=5.0,
                retreating=False,
                sourceX=None,
                sourceY=None,
            ),
            AttackSnapshot(
                id="b",
                attackerSmallID=1,
                targetSmallID=TERRA_NULLIUS_SMALL_ID,
                troops=5.0,
                retreating=False,
                sourceX=None,
                sourceY=None,
            ),
        ]

        _, features = self._encode(attacks=attacks)

        assert self._feature(features, 1, "is_attacking_viewer") == 1.0
        assert self._feature(features, 0, "viewer_is_attacking") == 1.0
        assert self._feature(features, 1, "viewer_is_attacking") == 0.0

    def test_raises_for_unknown_viewer(self) -> None:
        header = _make_header([LAND], width=1, height=1)
        tick = _make_tick([0], players=[_player(2)])
        with pytest.raises(ValueError, match="no player"):
            encode_player_candidates(header, tick, viewer_small_id=1)


def _unit(
    unit_type: str,
    owner: int,
    x: int,
    y: int = 0,
    *,
    target: tuple[int, int] | None = None,
) -> UnitSnapshot:
    return UnitSnapshot(
        id=hash((unit_type, owner, x, y)) & 0xFFFF,
        type=unit_type,
        ownerSmallID=owner,
        x=x,
        y=y,
        targetX=target[0] if target else None,
        targetY=target[1] if target else None,
        health=100,
        maxHealth=100,
        troops=0,
        level=1,
        isUnderConstruction=False,
    )


def _attack(attacker: int, target: int, troops: float) -> AttackSnapshot:
    return AttackSnapshot(
        id=f"{attacker}->{target}",
        attackerSmallID=attacker,
        targetSmallID=target,
        troops=troops,
        retreating=False,
        sourceX=None,
        sourceY=None,
    )


class TestSituationalAwareness:
    """The v3 features: where other players are, how strong they are, what
    they've built, and the attacks in progress.

    A 1x5 all-land strip owned [unowned, viewer(1), enemy(2), ally(3),
    enemy(4)]. The viewer has 100 troops; player 2 has 1000 (stronger), 3
    has 100 (similar), 4 has 10 (weaker)."""

    def _header(self) -> SnapshotHeader:
        return _make_header([LAND] * 5, width=5, height=1)

    def _tick(self, **overrides: object) -> TickSnapshot:
        tick = _make_tick(
            [0, 1, 2, 3, 4],
            players=[
                _player(1, troops=100.0, numTilesOwned=1),
                _player(2, troops=1000.0, numTilesOwned=1, type="NATION", clientID=None),
                _player(3, troops=100.0, numTilesOwned=1),
                _player(4, troops=10.0, numTilesOwned=1, type="BOT", clientID=None),
            ],
            alliances=[AllianceSnapshot(requestorSmallID=1, recipientSmallID=3, expiresAtTick=999)],
        )
        return tick.model_copy(update=overrides)

    def _channel(self, tick: TickSnapshot, name: str) -> list[int]:
        spatial = encode_spatial(self._header(), tick, viewer_small_id=1)
        return spatial[SPATIAL_CHANNELS.index(name)].flatten().tolist()

    # -- spatial -----------------------------------------------------------

    def test_every_channel_is_binary_uint8(self) -> None:
        tick = self._tick(
            units=[_unit("City", 1, 1), _unit("Atom Bomb", 2, 2, target=(1, 0))],
            attacks=[_attack(2, 1, 50.0)],
        )
        spatial = encode_spatial(self._header(), tick, viewer_small_id=1)

        assert spatial.dtype == np.uint8
        assert spatial.shape == (len(SPATIAL_CHANNELS), 1, 5)
        assert set(np.unique(spatial)) <= {0, 1}

    def test_owner_strength_buckets(self) -> None:
        tick = self._tick()

        assert self._channel(tick, "owner_stronger") == [0, 0, 1, 0, 0]
        assert self._channel(tick, "owner_similar") == [0, 0, 0, 1, 0]
        assert self._channel(tick, "owner_weaker") == [0, 0, 0, 0, 1]

    def test_viewer_and_unowned_land_get_no_strength(self) -> None:
        tick = self._tick()
        for name in ("owner_stronger", "owner_similar", "owner_weaker"):
            values = self._channel(tick, name)
            assert values[0] == 0 and values[1] == 0

    def test_attack_channels_mark_territory_not_unowned_land(self) -> None:
        tick = self._tick(
            attacks=[_attack(2, 1, 50.0), _attack(1, 4, 20.0), _attack(1, 0, 30.0)]
        )

        assert self._channel(tick, "owner_attacking_viewer") == [0, 0, 1, 0, 0]
        # Expanding into unowned land (target 0) doesn't mark anything.
        assert self._channel(tick, "owner_attacked_by_viewer") == [0, 0, 0, 0, 1]

    def test_unit_channels(self) -> None:
        tick = self._tick(
            units=[
                _unit("City", 1, 1),
                _unit("Transport", 1, 1),
                _unit("SAM Launcher", 2, 2),
                _unit("Port", 3, 3),  # ally: not an enemy structure
                _unit("Warship", 4, 4),
                _unit("Hydrogen Bomb", 2, 2, target=(1, 0)),
                _unit("Trade Ship", 2, 2),  # ignored entirely
            ]
        )

        assert self._channel(tick, "own_city") == [0, 1, 0, 0, 0]
        assert self._channel(tick, "own_port") == [0, 0, 0, 0, 0]
        assert self._channel(tick, "own_transport") == [0, 1, 0, 0, 0]
        assert self._channel(tick, "enemy_structure") == [0, 0, 1, 0, 0]
        assert self._channel(tick, "enemy_sam_launcher") == [0, 0, 1, 0, 0]
        assert self._channel(tick, "enemy_warship") == [0, 0, 0, 0, 1]
        assert self._channel(tick, "incoming_nuke_target") == [0, 1, 0, 0, 0]

    # -- flat ----------------------------------------------------------------

    def test_flat_attack_and_threat_totals(self) -> None:
        tick = self._tick(
            attacks=[_attack(2, 1, 50.0), _attack(4, 1, 30.0), _attack(1, 0, 70.0)],
            units=[
                _unit("Transport", 2, 2),
                _unit("Transport", 3, 3),  # ally boat: not a threat
                _unit("Atom Bomb", 4, 4, target=(1, 0)),
            ],
        )
        values = dict(zip(FLAT_FEATURES, encode_flat(tick, viewer_small_id=1), strict=True))

        assert values["incoming_attack_troops_log1p"] == pytest.approx(np.log1p(80.0))
        assert values["outgoing_attack_troops_log1p"] == pytest.approx(np.log1p(70.0))
        assert values["num_incoming_attacks"] == 2.0
        assert values["enemy_transports_log1p"] == pytest.approx(np.log1p(1))
        assert values["enemy_nukes_in_flight"] == 1.0

    def test_flat_own_structure_counts(self) -> None:
        tick = self._tick(
            units=[_unit("City", 1, 1), _unit("City", 1, 1), _unit("Port", 2, 2)]
        )
        values = dict(zip(FLAT_FEATURES, encode_flat(tick, viewer_small_id=1), strict=True))

        assert values["own_city_count_log1p"] == pytest.approx(np.log1p(2))
        assert values["own_port_count_log1p"] == 0.0  # the Port belongs to player 2

    # -- player candidates ---------------------------------------------------

    def _player_features(self, tick: TickSnapshot) -> dict[int, dict[str, float]]:
        small_ids, features = encode_player_candidates(self._header(), tick, viewer_small_id=1)
        return {
            int(small_id): dict(zip(PLAYER_FEATURES, row.tolist(), strict=True))
            for small_id, row in zip(small_ids, features, strict=True)
        }

    def test_player_attack_troops_both_directions(self) -> None:
        tick = self._tick(attacks=[_attack(2, 1, 50.0), _attack(1, 2, 20.0), _attack(1, 0, 9.0)])
        by_id = self._player_features(tick)

        assert by_id[2]["incoming_attack_troops_log1p"] == pytest.approx(np.log1p(50.0))
        assert by_id[2]["outgoing_attack_troops_log1p"] == pytest.approx(np.log1p(20.0))
        assert by_id[0]["outgoing_attack_troops_log1p"] == pytest.approx(np.log1p(9.0))
        assert by_id[4]["incoming_attack_troops_log1p"] == 0.0

    def test_centroids_are_relative_to_viewer_and_scaled_by_map_size(self) -> None:
        by_id = self._player_features(self._tick())

        # Viewer at x=1; player 2 at x=2, player 4 at x=4; map width 5.
        assert by_id[2]["centroid_dx"] == pytest.approx(1 / 5)
        assert by_id[4]["centroid_dx"] == pytest.approx(3 / 5)
        assert by_id[4]["centroid_distance"] == pytest.approx(3 / 5)
        assert by_id[2]["centroid_dy"] == 0.0
        # Terra nullius: the center of unowned land (x=0), left of the viewer.
        assert by_id[0]["centroid_dx"] == pytest.approx(-1 / 5)

    def test_player_type_and_structures(self) -> None:
        tick = self._tick(
            units=[_unit("City", 2, 2), _unit("SAM Launcher", 2, 2), _unit("Warship", 4, 4)]
        )
        by_id = self._player_features(tick)

        assert by_id[2]["is_nation"] == 1.0 and by_id[2]["is_bot"] == 0.0
        assert by_id[4]["is_bot"] == 1.0 and by_id[4]["is_nation"] == 0.0
        assert by_id[2]["structure_count_log1p"] == pytest.approx(np.log1p(2))
        assert by_id[2]["has_sam_launcher"] == 1.0
        # Warships aren't structures.
        assert by_id[4]["structure_count_log1p"] == 0.0

    def test_troops_per_tile(self) -> None:
        by_id = self._player_features(self._tick())

        assert by_id[2]["troops_per_tile_log1p"] == pytest.approx(np.log1p(1000.0))


class TestEncodeCandidateCells:
    """An 8x8 all-land map with cell_factor 4 -> a 2x2 cell grid. Owners are
    painted per test; candidates are [terra nullius, 2, 3] (viewer is 1)."""

    CANDIDATES = np.array([TERRA_NULLIUS_SMALL_ID, 2, 3], dtype=np.int64)

    def _cells(self, owners: np.ndarray, *, land: np.ndarray | None = None) -> list[list[int]]:
        height, width = owners.shape
        is_land = land if land is not None else np.ones_like(owners, dtype=bool)
        terrain = np.where(is_land, LAND, 0).astype(np.uint8)
        header = _make_header(terrain.flatten().tolist(), width=width, height=height)
        tick = _make_tick(
            owners.flatten().tolist(), players=[_player(1), _player(2), _player(3)]
        )
        cells = encode_candidate_cells(header, tick, 1, self.CANDIDATES, cell_factor=4)
        assert cells.dtype == np.int16
        return cells.tolist()

    def test_each_cell_belongs_to_its_majority_owner(self) -> None:
        owners = np.zeros((8, 8), dtype=np.int64)
        owners[:4, :4] = 2  # top-left: player 2 -> candidate index 1
        owners[4:, 4:] = 3  # bottom-right: player 3 -> candidate index 2
        # top-right and bottom-left stay unowned land -> candidate 0

        assert self._cells(owners) == [[1, 0], [0, 2]]

    def test_viewer_tiles_dont_decide_a_cell(self) -> None:
        """A front-line cell that's mostly the viewer's belongs to the
        neighbor in it, not to the viewer (who isn't a candidate)."""
        owners = np.zeros((8, 8), dtype=np.int64)
        owners[:4, :4] = 1  # top-left cell: all viewer...
        owners[0, 0] = 2  # ...except one tile of player 2
        assert self._cells(owners)[0][0] == 1

    def test_water_doesnt_decide_a_cell(self) -> None:
        owners = np.zeros((8, 8), dtype=np.int64)
        land = np.zeros((8, 8), dtype=bool)
        land[0, 0] = True  # top-left cell: one land tile, owned by player 3
        owners[0, 0] = 3
        cells = self._cells(owners, land=land)
        assert cells[0][0] == 2
        # Cells with no land at all have no owner.
        assert cells[1][1] == -1

    def test_cell_of_only_viewer_land_has_no_owner(self) -> None:
        owners = np.ones((8, 8), dtype=np.int64)
        assert self._cells(owners) == [[-1, -1], [-1, -1]]

    def test_grid_floors_like_the_conv_trunk(self) -> None:
        header = _make_header([LAND] * (9 * 10), width=10, height=9)
        tick = _make_tick([0] * (9 * 10), players=[_player(1)])
        cells = encode_candidate_cells(header, tick, 1, self.CANDIDATES, cell_factor=4)
        assert cells.shape == (2, 2)


class TestEncodeObservation:
    def test_returns_both_tensors(self) -> None:
        header = _make_header([0, 0], width=2, height=1)
        tick = _make_tick([1, 0], players=[_player(1)])

        spatial, flat = encode_observation(header, tick, viewer_small_id=1)

        assert spatial.shape == (len(SPATIAL_CHANNELS), 1, 2)
        assert flat.shape == (len(FLAT_FEATURES),)
