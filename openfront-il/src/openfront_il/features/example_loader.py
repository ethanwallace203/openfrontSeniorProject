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

**Memory:** snapshot ticks are read from disk on demand, never held in
memory. Construction scans each game's snapshot file once to index where
every tick's line starts (``replay.loader.index_snapshots``); ``example()``
then reads and parses just that one line. Memory is the processed rows
plus a few bytes per tick, independent of snapshot size — loading whole
games instead was measured at ~5x their on-disk size, which a 100-game
dataset (~33GB of snapshots) can't fit. The OS page cache keeps recently
read lines fast, and is shared by every DataLoader worker for free.
"""

from __future__ import annotations

from typing import BinaryIO

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from openfront_il.features.action_encoder import (
    ACTION_TYPES,
    PLAYER_TARGET_ACTION_TYPES,
    TILE_TARGET_ACTION_TYPES,
    encode_player_target_mask,
    encode_tile_target_mask,
)
from openfront_il.features.observation_encoder import (
    FLAT_FEATURES,
    encode_player_candidates,
    encode_spatial,
)
from openfront_il.replay import loader
from openfront_il.replay.snapshot_schema import TickSnapshot
from openfront_il.utils import paths


class ExampleTable:
    """Loads several games' processed rows plus a byte-offset index into
    each game's snapshot file, so ``example()`` can read back exactly the
    one tick a row needs (see the module docstring).

    Snapshot files are opened lazily, once per process. Reads go through
    ``os.pread``, so a file inherited across a fork is safe to share between
    DataLoader workers; pickling (spawned workers on Windows/macOS) drops
    the open files and each worker reopens its own.
    """

    def __init__(self, game_ids: list[str]) -> None:
        self.game_ids = game_ids
        tables = [
            pq.read_table(paths.PROCESSED_DIR / f"{game_id}.parquet").to_pandas()
            for game_id in game_ids
        ]
        self.rows: pd.DataFrame = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

        self._indexes = {game_id: loader.index_snapshots(game_id) for game_id in game_ids}
        self._files: dict[str, BinaryIO] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def map_shape(self, game_id: str) -> tuple[int, int]:
        """``(height, width)`` of a game's map — every example from that
        game has a spatial tensor of exactly this size."""
        header = self._indexes[game_id].header
        return header.height, header.width

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_files"] = {}  # open files don't pickle; reopened lazily
        return state

    def close(self) -> None:
        """Closes any snapshot files this process opened. Safe to call more
        than once; garbage collection calls it too, as a fallback."""
        for file in self._files.values():
            file.close()
        self._files.clear()

    def __del__(self) -> None:
        # getattr: __init__ may have failed before _files existed.
        if getattr(self, "_files", None):
            self.close()

    def _tick(self, game_id: str, tick: int) -> TickSnapshot:
        file = self._files.get(game_id)
        if file is None:
            file = paths.snapshot_path(game_id).open("rb", buffering=0)
            self._files[game_id] = file
        offset, length = self._indexes[game_id].tick_lines[tick]
        return loader.read_tick(file, offset, length)

    def example(self, index: int) -> dict[str, np.ndarray | np.int64]:
        """Returns one example as plain numpy arrays:

        - ``spatial`` ``[C, H, W]``, ``flat`` ``[F]``, ``action_type`` and
          ``unit_type`` (0-d int64), ``legality_mask`` ``[len(ACTION_TYPES)]``.
        - ``target_x`` / ``target_y`` (0-d int64): the tile argument, or
          ``-1`` when this row's action type takes no tile (or its tile
          couldn't be decoded). ``tile_legal`` ``[H, W]`` bool.
        - ``player_features`` ``[N, P]``, ``player_legal`` ``[N]`` bool,
          and ``target_player`` (0-d int64): the index of the target among
          those ``N`` candidates, or ``-1`` when this row's action type
          takes no player (or the target isn't a candidate, e.g. a player
          who died during the snapshot window).
        """
        row = self.rows.iloc[index]
        game_id = row["game_id"]
        header = self._indexes[game_id].header
        tick = self._tick(game_id, int(row["tick"]))
        small_id = int(row["small_id"])
        action_type = int(row["action_type"])
        unit_type = int(row["unit_type"])
        action_name = ACTION_TYPES[action_type]

        spatial = encode_spatial(header, tick, viewer_small_id=small_id)
        flat = row[FLAT_FEATURES].to_numpy(dtype=np.float32)
        legality_mask = np.asarray(row["legality_mask"], dtype=bool)

        target_x = target_y = -1
        if action_name in TILE_TARGET_ACTION_TYPES and not (
            pd.isna(row["target_x"]) or pd.isna(row["target_y"])
        ):
            target_x, target_y = int(row["target_x"]), int(row["target_y"])

        candidate_ids, player_features = encode_player_candidates(header, tick, small_id)
        player_legal = encode_player_target_mask(
            tick.legal[row["client_id"]], candidate_ids, action_type
        )
        target_player = -1
        if action_name in PLAYER_TARGET_ACTION_TYPES and not pd.isna(row["target_player_small_id"]):
            matches = np.flatnonzero(candidate_ids == int(row["target_player_small_id"]))
            if len(matches) > 0:
                target_player = int(matches[0])

        return {
            "spatial": spatial,
            "flat": flat,
            "action_type": np.int64(action_type),
            "unit_type": np.int64(unit_type),
            "legality_mask": legality_mask,
            "target_x": np.int64(target_x),
            "target_y": np.int64(target_y),
            "tile_legal": encode_tile_target_mask(spatial, action_type, unit_type),
            "player_features": player_features,
            "player_legal": player_legal,
            "target_player": np.int64(target_player),
        }
