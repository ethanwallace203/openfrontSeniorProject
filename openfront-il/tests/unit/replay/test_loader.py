import shutil
from pathlib import Path

import pytest

from openfront_il.replay import loader
from openfront_il.replay.snapshot_schema import SnapshotHeader, TickSnapshot
from openfront_il.utils import paths

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "USR3SJiv.snapshots.jsonl"


@pytest.fixture(autouse=True)
def _snapshot_fixture_in_place() -> None:
    paths.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE, paths.snapshot_path("USR3SJiv"))
    yield
    shutil.rmtree(paths.SNAPSHOTS_DIR, ignore_errors=True)


def test_iter_snapshots_yields_header_then_ticks() -> None:
    lines = list(loader.iter_snapshots("USR3SJiv"))

    assert isinstance(lines[0], SnapshotHeader)
    assert all(isinstance(line, TickSnapshot) for line in lines[1:])
    assert len(lines) > 1


def test_iter_snapshots_missing_game_raises() -> None:
    with pytest.raises(FileNotFoundError):
        list(loader.iter_snapshots("doesNotExist"))


def test_load_snapshots_splits_header_and_ticks() -> None:
    header, ticks = loader.load_snapshots("USR3SJiv")

    assert isinstance(header, SnapshotHeader)
    assert header.gameId == "USR3SJiv"
    assert len(ticks) > 0
    assert all(isinstance(tick, TickSnapshot) for tick in ticks)
    # Ticks should be in increasing order.
    assert [t.tick for t in ticks] == sorted(t.tick for t in ticks)


def test_iter_snapshots_is_lazy() -> None:
    """Pulling the header should not eagerly validate the rest of the
    file — a real snapshot file can be hundreds of MB to low GB, so this
    module's whole reason to exist is not reading it all into memory."""
    header_line = FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    corrupted = f'{header_line}\n{{"kind": "tick", "this is": "not a valid tick"}}\n'
    paths.snapshot_path("USR3SJiv").write_text(corrupted, encoding="utf-8")

    gen = loader.iter_snapshots("USR3SJiv")
    first = next(gen)  # must not raise, even though the next line is invalid
    assert isinstance(first, SnapshotHeader)

    with pytest.raises(Exception):  # noqa: B017 - pydantic's ValidationError
        next(gen)
