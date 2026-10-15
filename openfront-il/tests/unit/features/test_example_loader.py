import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest

from openfront_il.features import dataset_builder
from openfront_il.features.action_encoder import ACTION_TYPES
from openfront_il.features.example_loader import ExampleTable
from openfront_il.features.observation_encoder import (
    FLAT_FEATURES,
    PLAYER_FEATURES,
    SPATIAL_CHANNELS,
)
from openfront_il.utils import paths

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "USR3SJiv.snapshots.jsonl"


@pytest.fixture
def _processed_fixture() -> None:
    paths.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE, paths.snapshot_path("USR3SJiv"))
    dataset_builder.build_dataset_for_game("USR3SJiv")
    yield
    shutil.rmtree(paths.SNAPSHOTS_DIR, ignore_errors=True)
    shutil.rmtree(paths.PROCESSED_DIR, ignore_errors=True)


class TestExampleTable:
    def test_len_matches_parquet_row_count(self, _processed_fixture: None) -> None:
        table = ExampleTable(["USR3SJiv"])
        assert len(table) == len(table.rows)
        assert len(table) > 0

    def test_example_shapes_and_dtypes(self, _processed_fixture: None) -> None:
        table = ExampleTable(["USR3SJiv"])

        example = table.example(0)

        assert example["spatial"].shape[0] == len(SPATIAL_CHANNELS)
        assert example["spatial"].ndim == 3
        assert example["flat"].shape == (len(FLAT_FEATURES),)
        assert example["flat"].dtype == np.float32
        assert example["legality_mask"].shape == (len(ACTION_TYPES),)
        assert example["legality_mask"].dtype == bool
        assert isinstance(example["action_type"], np.int64)
        assert example["tile_legal"].shape == example["spatial"].shape[1:]
        assert example["tile_legal"].dtype == bool
        candidates = example["player_features"].shape[0]
        assert example["player_features"].shape == (candidates, len(PLAYER_FEATURES))
        assert example["player_legal"].shape == (candidates,)

    def test_spawn_rows_carry_an_in_bounds_tile_target(self, _processed_fixture: None) -> None:
        table = ExampleTable(["USR3SJiv"])
        spawn = ACTION_TYPES.index("spawn")

        examples = [table.example(i) for i in range(len(table))]
        spawns = [example for example in examples if example["action_type"] == spawn]

        assert len(spawns) == 2
        for example in spawns:
            height, width = example["spatial"].shape[1:]
            assert 0 <= example["target_x"] < width
            assert 0 <= example["target_y"] < height
            # Never a player argument for a tile action.
            assert example["target_player"] == -1

    def test_non_argument_rows_have_no_targets(self, _processed_fixture: None) -> None:
        table = ExampleTable(["USR3SJiv"])
        noop = ACTION_TYPES.index("noop")

        example = next(
            table.example(i) for i in range(len(table)) if table.rows.iloc[i]["action_type"] == noop
        )

        assert example["target_x"] == -1
        assert example["target_player"] == -1
        assert not example["tile_legal"].any()
        assert not example["player_legal"].any()

    def test_spatial_is_one_byte_per_value(self, _processed_fixture: None) -> None:
        example = ExampleTable(["USR3SJiv"]).example(0)

        assert example["spatial"].dtype == np.uint8
        assert set(np.unique(example["spatial"])) <= {0, 1}

    def test_keeps_only_an_index_not_the_snapshots(self, _processed_fixture: None) -> None:
        """The whole point: memory must not scale with snapshot size."""
        table = ExampleTable(["USR3SJiv"])

        index = table._indexes["USR3SJiv"]
        assert all(
            isinstance(offset, int) and isinstance(length, int)
            for offset, length in index.tick_lines.values()
        )
        assert not hasattr(table, "_ticks_by_game")

    def test_survives_pickling_for_spawned_workers(self, _processed_fixture: None) -> None:
        table = ExampleTable(["USR3SJiv"])
        original = table.example(0)  # opens a snapshot file in this process

        restored = pickle.loads(pickle.dumps(table))

        assert restored._files == {}
        np.testing.assert_array_equal(restored.example(0)["spatial"], original["spatial"])
        table.close()
        restored.close()

    def test_empty_game_list_gives_empty_table(self) -> None:
        table = ExampleTable([])
        assert len(table) == 0
