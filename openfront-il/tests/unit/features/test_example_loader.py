import shutil
from pathlib import Path

import numpy as np
import pytest

from openfront_il.features import dataset_builder
from openfront_il.features.action_encoder import ACTION_TYPES
from openfront_il.features.example_loader import ExampleTable
from openfront_il.features.observation_encoder import FLAT_FEATURES, SPATIAL_CHANNELS
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

    def test_empty_game_list_gives_empty_table(self) -> None:
        table = ExampleTable([])
        assert len(table) == 0
