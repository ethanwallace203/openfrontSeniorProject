"""Reads back (state, action) examples from ``data/processed/*.parquet``,
reconstructing each example's spatial tensor on demand from
``data/snapshots/`` (see ``dataset_builder.py``'s module docstring for why
spatial tensors aren't stored in the parquet files directly).

Framework-agnostic (plain numpy, no torch) so both ``training/``'s
``torch.utils.data.Dataset`` wrapper and ``evaluation/``'s inference loop
build on exactly one implementation of "load processed rows + rebuild
spatial" instead of two duplicated (and potentially drifting) copies. This
is why the logic lives here rather than in `training/` or `evaluation/`
directly — both are allowed to import `features/`, neither is allowed to
import the other (see their READMEs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from openfront_il.features.observation_encoder import FLAT_FEATURES, encode_spatial
from openfront_il.replay import loader
from openfront_il.replay.snapshot_schema import SnapshotHeader, TickSnapshot
from openfront_il.utils import paths


class ExampleTable:
    """Loads several games' processed rows plus the snapshot data needed
    to reconstruct each row's spatial tensor.

    **Scale note:** loads every referenced game's full snapshot file into
    memory up front, to support random-access ``example()`` calls without
    re-scanning a JSONL file each time. Fine for a handful of small/medium
    games; a real ~100-game fetch run with some multi-GB snapshot files
    (see ``engine/bridge/README.md``'s measured storage finding) would
    need a proper on-disk random-access index instead — not built here,
    per "clarity over performance."
    """

    def __init__(self, game_ids: list[str]) -> None:
        self.game_ids = game_ids
        tables = [
            pq.read_table(paths.PROCESSED_DIR / f"{game_id}.parquet").to_pandas()
            for game_id in game_ids
        ]
        self.rows: pd.DataFrame = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

        self._headers: dict[str, SnapshotHeader] = {}
        self._ticks_by_game: dict[str, dict[int, TickSnapshot]] = {}
        for game_id in game_ids:
            header, ticks = loader.load_snapshots(game_id)
            self._headers[game_id] = header
            self._ticks_by_game[game_id] = {tick.tick: tick for tick in ticks}

    def __len__(self) -> int:
        return len(self.rows)

    def example(self, index: int) -> dict[str, np.ndarray | np.int64]:
        """Returns one example as plain numpy arrays: ``spatial``,
        ``flat``, ``action_type`` (0-d int64), ``unit_type`` (0-d int64),
        ``legality_mask``.
        """
        row = self.rows.iloc[index]
        game_id = row["game_id"]
        header = self._headers[game_id]
        tick = self._ticks_by_game[game_id][int(row["tick"])]

        spatial = encode_spatial(header, tick, viewer_small_id=int(row["small_id"]))
        flat = row[FLAT_FEATURES].to_numpy(dtype=np.float32)
        legality_mask = np.asarray(row["legality_mask"], dtype=bool)

        return {
            "spatial": spatial,
            "flat": flat,
            "action_type": np.int64(row["action_type"]),
            "unit_type": np.int64(row["unit_type"]),
            "legality_mask": legality_mask,
        }
