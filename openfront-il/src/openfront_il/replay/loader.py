"""Streaming reader for ``data/snapshots/{gameId}.jsonl`` files.

Reads one line at a time so a caller never has to hold an entire game's
snapshot file in memory — real games can produce files in the hundreds of
MB to low GB range (measured; see ``engine/bridge/README.md``'s storage
finding).
"""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import TypeAdapter

from openfront_il.replay.snapshot_schema import SnapshotHeader, SnapshotLine, TickSnapshot
from openfront_il.utils import paths

_line_adapter: TypeAdapter[SnapshotHeader | TickSnapshot] = TypeAdapter(SnapshotLine)


def iter_snapshots(game_id: str) -> Iterator[SnapshotHeader | TickSnapshot]:
    """Yields each validated line of a game's snapshot file, in order: the
    ``SnapshotHeader`` first, then one ``TickSnapshot`` per subsequent
    line.

    Raises:
        FileNotFoundError: if the game has no snapshot file.
    """
    with paths.snapshot_path(game_id).open("r", encoding="utf-8") as file:
        for raw_line in file:
            stripped = raw_line.strip()
            if not stripped:
                continue
            yield _line_adapter.validate_json(stripped)


def load_snapshots(game_id: str) -> tuple[SnapshotHeader, list[TickSnapshot]]:
    """Loads a full game's snapshots into memory.

    Convenience for small fixtures and tests — prefer `iter_snapshots` for
    real games, whose files can be hundreds of MB to low GB.

    Raises:
        FileNotFoundError: if the game has no snapshot file.
        ValueError: if the file doesn't start with a header, or a
            non-tick line appears after it.
    """
    lines = iter_snapshots(game_id)
    header = next(lines)
    if not isinstance(header, SnapshotHeader):
        raise ValueError(f"expected the first line of {game_id}'s snapshot file to be a header")

    ticks: list[TickSnapshot] = []
    for line in lines:
        if not isinstance(line, TickSnapshot):
            raise ValueError(f"expected a tick snapshot, got {type(line).__name__}")
        ticks.append(line)
    return header, ticks
