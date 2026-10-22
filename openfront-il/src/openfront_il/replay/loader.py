"""Streaming reader for ``data/snapshots/{gameId}.jsonl`` files.

Reads one line at a time so a caller never has to hold an entire game's
snapshot file in memory — real games can produce files in the hundreds of
MB to low GB range (measured; see ``engine/bridge/README.md``'s storage
finding).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import BinaryIO

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


# Every tick line starts `{"kind":"tick","tick":N,` (key order is fixed by
# engine/bridge/src/snapshotWriter.ts), so the tick number can be read off
# the first few bytes without parsing a multi-hundred-KB line.
_TICK_LINE_PREFIX = re.compile(rb'^\s*\{"kind":"tick","tick":(\d+)[,}]')
_PREFIX_BYTES = 64


@dataclass(frozen=True)
class SnapshotIndex:
    """Where each tick's line lives in a game's snapshot file, so a single
    tick can be read back without loading the rest of the file."""

    header: SnapshotHeader
    # tick -> (byte offset, byte length) of that tick's JSONL line.
    tick_lines: dict[int, tuple[int, int]]


def index_snapshots(game_id: str) -> SnapshotIndex:
    """Scans a game's snapshot file once, validating only the header and
    recording each tick line's byte range. Memory is one small tuple per
    tick, independent of how large the lines themselves are.

    A line whose prefix doesn't match the expected tick layout is fully
    parsed instead (slower, but never silently mis-indexed).

    Raises:
        FileNotFoundError: if the game has no snapshot file.
        ValueError: if the file doesn't start with a header, or a
            non-tick line appears after it.
    """
    header: SnapshotHeader | None = None
    tick_lines: dict[int, tuple[int, int]] = {}
    offset = 0
    with paths.snapshot_path(game_id).open("rb") as file:
        for raw_line in file:
            length = len(raw_line)
            if raw_line.strip():
                if header is None:
                    parsed = _line_adapter.validate_json(raw_line)
                    if not isinstance(parsed, SnapshotHeader):
                        raise ValueError(
                            f"expected the first line of {game_id}'s snapshot file to be a header"
                        )
                    header = parsed
                else:
                    tick_lines[_tick_number(raw_line)] = (offset, length)
            offset += length
    if header is None:
        raise ValueError(f"{game_id}'s snapshot file is empty")
    return SnapshotIndex(header=header, tick_lines=tick_lines)


def _tick_number(raw_line: bytes) -> int:
    match = _TICK_LINE_PREFIX.match(raw_line[:_PREFIX_BYTES])
    if match is not None:
        return int(match.group(1))
    return _parse_tick(raw_line).tick


def _parse_tick(raw_line: bytes) -> TickSnapshot:
    parsed = _line_adapter.validate_json(raw_line)
    if not isinstance(parsed, TickSnapshot):
        raise ValueError(f"expected a tick snapshot, got {type(parsed).__name__}")
    return parsed


def read_tick(file: BinaryIO, offset: int, length: int) -> TickSnapshot:
    """Reads and validates one tick line at a ``SnapshotIndex`` byte range.

    Uses ``os.pread`` where available: it reads at an absolute offset
    without moving the file position, so a file object inherited by forked
    DataLoader workers (which share one OS-level position) is safe to read
    from all of them concurrently. ``seek`` + ``read`` is the fallback on
    platforms without ``pread`` (Windows), where workers are spawned and
    each opens its own file anyway.
    """
    if hasattr(os, "pread"):
        raw_line = os.pread(file.fileno(), length, offset)
    else:
        file.seek(offset)
        raw_line = file.read(length)
    if len(raw_line) != length:
        raise ValueError(
            f"short read at offset {offset}: expected {length} bytes, got {len(raw_line)} "
            "(snapshot file changed since it was indexed?)"
        )
    return _parse_tick(raw_line)
