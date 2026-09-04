import base64
import gzip

import numpy as np
import pytest

from openfront_il.features.observation_encoder import (
    FLAT_FEATURES,
    SPATIAL_CHANNELS,
    encode_flat,
    encode_observation,
    encode_spatial,
)
from openfront_il.replay.snapshot_schema import (
    AllianceSnapshot,
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
        formatVersion=1,
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


class TestEncodeObservation:
    def test_returns_both_tensors(self) -> None:
        header = _make_header([0, 0], width=2, height=1)
        tick = _make_tick([1, 0], players=[_player(1)])

        spatial, flat = encode_observation(header, tick, viewer_small_id=1)

        assert spatial.shape == (len(SPATIAL_CHANNELS), 1, 2)
        assert flat.shape == (len(FLAT_FEATURES),)
