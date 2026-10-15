import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from openfront_il.features import dataset_builder
from openfront_il.features.action_encoder import ACTION_TYPES
from openfront_il.replay.snapshot_schema import LegalityMask, PlayerScalars, TickSnapshot
from openfront_il.utils import paths

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "USR3SJiv.snapshots.jsonl"


def _player(small_id: int, client_id: str | None, **overrides: object) -> PlayerScalars:
    defaults: dict[str, object] = dict(
        id=f"pid{small_id}",
        clientID=client_id,
        smallID=small_id,
        username=f"p{small_id}",
        type="HUMAN",
        team=None,
        isAlive=True,
        isTraitor=False,
        isDisconnected=False,
        numTilesOwned=1,
        gold="0",
        troops=0.0,
    )
    defaults.update(overrides)
    return PlayerScalars.model_validate(defaults)


def _legality() -> LegalityMask:
    return LegalityMask.model_validate(
        dict(
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
    )


def _tick(**overrides: object) -> TickSnapshot:
    defaults: dict[str, object] = dict(
        kind="tick",
        tick=10,
        spawnPhase=False,
        tileStateGzipBase64="",
        players=[],
        units=[],
        attacks=[],
        alliances=[],
        actions={},
        legal={},
        winner=None,
    )
    defaults.update(overrides)
    return TickSnapshot.model_validate(defaults)


class TestBuildRowsForTick:
    def test_noop_row_when_no_actions(self) -> None:
        tick = _tick(
            players=[_player(1, "alice")],
            legal={"alice": _legality()},
            actions={},
        )

        rows = dataset_builder.build_rows_for_tick(tick, game_id="g1", map_width=100)

        assert len(rows) == 1
        assert rows[0]["action_type"] == ACTION_TYPES.index("noop")
        assert rows[0]["client_id"] == "alice"
        assert rows[0]["game_id"] == "g1"
        assert rows[0]["tick"] == 10

    def test_one_row_per_modeled_action(self) -> None:
        tick = _tick(
            players=[_player(1, "alice"), _player(2, "bob")],
            legal={"alice": _legality()},
            actions={
                "alice": [
                    {"type": "attack", "targetID": "pid2", "troops": 10, "clientID": "alice"},
                    {"type": "donate_gold", "recipient": "pid2", "gold": 5, "clientID": "alice"},
                ]
            },
        )

        rows = dataset_builder.build_rows_for_tick(tick, game_id="g1", map_width=100)

        assert len(rows) == 2
        assert {row["action_type"] for row in rows} == {
            ACTION_TYPES.index("attack"),
            ACTION_TYPES.index("donate_gold"),
        }

    def test_targets_resolve_by_player_id_including_nations_and_tribes(self) -> None:
        """Nations and tribes have no clientID; humans' PlayerIDs differ from
        their clientIDs. Every target must still resolve to a smallID."""
        tick = _tick(
            players=[
                _player(1, "alice"),
                _player(2, "bob"),
                _player(3, None, type="NATION"),
                _player(4, None, type="BOT"),
            ],
            legal={"alice": _legality()},
            actions={
                "alice": [
                    {"type": "attack", "targetID": "pid2", "troops": 1, "clientID": "alice"},
                    {"type": "attack", "targetID": "pid3", "troops": 1, "clientID": "alice"},
                    {"type": "embargo", "targetID": "pid4", "action": "start", "clientID": "alice"},
                ]
            },
        )

        rows = dataset_builder.build_rows_for_tick(tick, game_id="g1", map_width=100)

        assert [row["target_player_small_id"] for row in rows] == [2, 3, 4]

    def test_unmodeled_only_actions_fall_back_to_noop(self) -> None:
        tick = _tick(
            players=[_player(1, "alice")],
            legal={"alice": _legality()},
            actions={
                "alice": [{"type": "emoji", "recipient": "bob", "emoji": 1, "clientID": "alice"}]
            },
        )

        rows = dataset_builder.build_rows_for_tick(tick, game_id="g1", map_width=100)

        assert len(rows) == 1
        assert rows[0]["action_type"] == ACTION_TYPES.index("noop")

    def test_skips_clients_with_no_matching_player(self) -> None:
        tick = _tick(players=[], legal={"ghost": _legality()})

        rows = dataset_builder.build_rows_for_tick(tick, game_id="g1", map_width=100)

        assert rows == []

    def test_tile_target_decoded_with_map_width(self) -> None:
        tick = _tick(
            players=[_player(1, "alice")],
            legal={"alice": _legality()},
            actions={"alice": [{"type": "spawn", "tile": 3 * 100 + 42, "clientID": "alice"}]},
        )

        [row] = dataset_builder.build_rows_for_tick(tick, game_id="g1", map_width=100)

        assert (row["target_x"], row["target_y"]) == (42, 3)

    def test_row_has_all_flat_feature_and_legality_columns(self) -> None:
        tick = _tick(players=[_player(1, "alice")], legal={"alice": _legality()})

        [row] = dataset_builder.build_rows_for_tick(tick, game_id="g1", map_width=100)

        assert "viewer_gold_log1p" in row
        assert len(row["legality_mask"]) == len(ACTION_TYPES)


@pytest.fixture
def _snapshots_dir() -> None:
    paths.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(paths.SNAPSHOTS_DIR, ignore_errors=True)
    shutil.rmtree(paths.PROCESSED_DIR, ignore_errors=True)


class TestBuildDatasetForGame:
    def test_builds_parquet_from_a_real_game(self, _snapshots_dir: None) -> None:
        shutil.copy(FIXTURE, paths.snapshot_path("USR3SJiv"))

        output_path = dataset_builder.build_dataset_for_game("USR3SJiv")

        assert output_path is not None
        assert output_path.exists()
        table = pq.read_table(output_path)
        assert table.num_rows > 0
        assert "action_type" in table.column_names
        assert "legality_mask" in table.column_names
        # The fixture's two spawns carry real tile targets inside the map.
        rows = table.to_pylist()
        spawns = [row for row in rows if row["action_type"] == ACTION_TYPES.index("spawn")]
        assert len(spawns) == 2
        for row in spawns:
            assert row["target_x"] is not None and row["target_x"] >= 0
            assert row["target_y"] is not None and row["target_y"] >= 0

    def test_raises_if_output_already_exists(self, _snapshots_dir: None) -> None:
        shutil.copy(FIXTURE, paths.snapshot_path("USR3SJiv"))
        dataset_builder.build_dataset_for_game("USR3SJiv")

        with pytest.raises(FileExistsError):
            dataset_builder.build_dataset_for_game("USR3SJiv")

    def test_empty_game_produces_no_file(self, _snapshots_dir: None) -> None:
        header_line = FIXTURE.read_text(encoding="utf-8").splitlines()[0]
        paths.snapshot_path("emptyGame").write_text(header_line + "\n", encoding="utf-8")

        result = dataset_builder.build_dataset_for_game("emptyGame")

        assert result is None
        assert not (paths.PROCESSED_DIR / "emptyGame.parquet").exists()


class TestBuildDataset:
    def test_discovers_and_skips_existing(self, _snapshots_dir: None) -> None:
        shutil.copy(FIXTURE, paths.snapshot_path("USR3SJiv"))

        first = dataset_builder.build_dataset()
        second = dataset_builder.build_dataset()

        assert first == second
        assert len(first) == 1
