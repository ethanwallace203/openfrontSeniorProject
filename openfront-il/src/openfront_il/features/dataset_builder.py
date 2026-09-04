"""Walks ``data/snapshots/*.jsonl`` (via ``replay/loader.py``), applies the
feature encoders, and writes ``data/processed/{gameId}.parquet``: one row
per (snapshot, acting human, action) example.

**Design note on spatial tensors: not stored here.**
``observation_encoder.encode_spatial()`` is a pure, cheap function of a
snapshot's already-compact tile-state buffer, so storing the decoded
``(C, H, W)`` float32 array *again* in ``data/processed/`` would multiply
an already measured storage problem (a single real game's snapshots can
run to 1GB+ — see ``engine/bridge/README.md``) for no benefit. Each row
instead carries a ``(game_id, tick)`` back-reference; a training-time
data loader re-derives the spatial tensor from
``data/snapshots/{gameId}.jsonl`` on demand via ``encode_spatial()``. This
is a real efficiency tradeoff for a prototype — re-seeking into a JSONL
file per example is not fast random access — and the fix, if it becomes a
bottleneck, is a precomputed line-offset index or a separate compact array
export, neither of which is built here, per "clarity over performance."
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from openfront_il.features.action_encoder import encode_action, encode_legality_mask
from openfront_il.features.observation_encoder import FLAT_FEATURES, encode_flat
from openfront_il.replay import loader
from openfront_il.replay.snapshot_schema import TickSnapshot
from openfront_il.utils import paths
from openfront_il.utils.logging import get_logger

logger = get_logger(__name__)


def _small_id_by_client_id(tick: TickSnapshot) -> dict[str, int]:
    return {
        player.clientID: player.smallID for player in tick.players if player.clientID is not None
    }


def build_rows_for_tick(tick: TickSnapshot, *, game_id: str) -> list[dict[str, object]]:
    """Pure function: one ``TickSnapshot`` -> its (state, action) example
    rows — one per living human with a ``legal`` entry, one row per
    modeled action they took during this window, or a single noop row if
    they took none (or only unmodeled actions — see
    ``action_encoder.UNMODELED_INTENT_TYPES``).

    Multiple modeled actions in one window each become their own row,
    sharing the same state features — this snapshot cadence (every
    ``snapshotEvery`` ticks) doesn't have per-intent-granularity state, so
    that's the finest resolution available; see
    ``docs/imitation_learning_plan.md`` Section 2.
    """
    rows: list[dict[str, object]] = []
    small_id_by_client = _small_id_by_client_id(tick)
    player_by_small_id = {player.smallID: player for player in tick.players}

    for client_id, legality in tick.legal.items():
        small_id = small_id_by_client.get(client_id)
        if small_id is None:
            continue
        player = player_by_small_id.get(small_id)
        if player is None:
            continue

        flat = encode_flat(tick, viewer_small_id=small_id)
        mask = encode_legality_mask(
            legality, spawn_phase=tick.spawnPhase, viewer_num_tiles_owned=player.numTilesOwned
        )

        raw_intents = tick.actions.get(client_id, [])
        encoded = [
            action
            for intent in raw_intents
            if (action := encode_action(intent, small_id_by_client_id=small_id_by_client))
            is not None
        ]
        if not encoded:
            noop = encode_action(None, small_id_by_client_id=small_id_by_client)
            assert noop is not None  # encode_action(None, ...) always returns NOOP_ACTION
            encoded = [noop]

        for action in encoded:
            rows.append(
                {
                    "game_id": game_id,
                    "tick": tick.tick,
                    "client_id": client_id,
                    "small_id": small_id,
                    **dict(zip(FLAT_FEATURES, flat.tolist(), strict=True)),
                    "action_type": action.action_type,
                    "unit_type": action.unit_type,
                    "target_player_small_id": action.target_player_small_id,
                    "target_x": action.target_x,
                    "target_y": action.target_y,
                    "amount": action.amount,
                    "legality_mask": mask.tolist(),
                }
            )
    return rows


def build_dataset_for_game(game_id: str) -> Path | None:
    """Streams one game's snapshots, builds example rows, and writes
    ``data/processed/{game_id}.parquet``.

    Returns:
        The output path, or ``None`` if the game produced zero rows (e.g.
        no living human ever had a ``legal`` entry — an aborted or
        near-empty game) — nothing is written in that case.

    Raises:
        FileExistsError: if ``data/processed/{game_id}.parquet`` already
            exists (immutability policy — see ``data/README.md``).
    """
    output_path = paths.PROCESSED_DIR / f"{game_id}.parquet"
    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists; delete it first to regenerate")

    rows: list[dict[str, object]] = []
    for line in loader.iter_snapshots(game_id):
        if isinstance(line, TickSnapshot):
            rows.extend(build_rows_for_tick(line, game_id=game_id))

    if not rows:
        logger.warning("%s produced zero rows; nothing written", game_id)
        return None

    paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    tmp_path = output_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp_path)
    tmp_path.rename(output_path)
    logger.info("wrote %d rows for %s to %s", len(rows), game_id, output_path)
    return output_path


def build_dataset(game_ids: list[str] | None = None) -> list[Path]:
    """Builds ``data/processed/*.parquet`` for every game in
    ``data/snapshots/`` (or just ``game_ids`` if given) that doesn't
    already have a processed file.
    """
    if game_ids is None:
        game_ids = sorted(path.stem for path in paths.SNAPSHOTS_DIR.glob("*.jsonl"))

    outputs: list[Path] = []
    for game_id in game_ids:
        output_path = paths.PROCESSED_DIR / f"{game_id}.parquet"
        if output_path.exists():
            logger.info("skipping %s: %s already exists", game_id, output_path)
            outputs.append(output_path)
            continue
        built = build_dataset_for_game(game_id)
        if built is not None:
            outputs.append(built)
    return outputs
