"""Drift alarm: validates a real snapshot file (produced by
engine/bridge/src/snapshotWriter.ts) against this project's Python mirror
of that schema. See snapshot_schema.py's module docstring for the
hand-sync contract this test exists to guard.
"""

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from openfront_il.replay.snapshot_schema import SnapshotHeader, SnapshotLine, TickSnapshot

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "USR3SJiv.snapshots.jsonl"

_line_adapter: TypeAdapter[SnapshotHeader | TickSnapshot] = TypeAdapter(SnapshotLine)


def _raw_lines() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_fixture_has_a_header_first() -> None:
    lines = _raw_lines()
    assert lines[0]["kind"] == "header"
    assert all(line["kind"] == "tick" for line in lines[1:])


def test_every_real_line_validates() -> None:
    for raw_line in _raw_lines():
        # Round-trips through JSON text since validate_json is what
        # loader.py actually uses (TypeAdapter.validate_python has
        # slightly different coercion rules).
        _line_adapter.validate_json(json.dumps(raw_line))


def test_header_fields() -> None:
    header = _line_adapter.validate_json(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(header, SnapshotHeader)
    assert header.gameId == "USR3SJiv"
    assert header.width > 0 and header.height > 0
    assert header.snapshotEvery == 10


def test_tick_snapshot_fields() -> None:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    tick = _line_adapter.validate_json(lines[1])
    assert isinstance(tick, TickSnapshot)
    assert tick.tick > 0
    assert len(tick.players) >= 1


def test_unknown_field_fails_loudly() -> None:
    """extra="forbid" is deliberate for this schema (unlike api_contract's
    extra="allow") — a field types.ts adds without updating this file
    should fail here, not be silently swallowed."""
    lines = _raw_lines()
    tick_line = next(line for line in lines if line["kind"] == "tick")
    tick_line["aBrandNewFieldNobodyMirroredYet"] = True
    with pytest.raises(Exception):  # noqa: B017 - pydantic's ValidationError
        _line_adapter.validate_json(json.dumps(tick_line))
