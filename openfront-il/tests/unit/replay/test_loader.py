import os
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


class TestIndexSnapshots:
    def test_indexes_every_tick_with_header(self) -> None:
        index = loader.index_snapshots("USR3SJiv")
        _, ticks = loader.load_snapshots("USR3SJiv")

        assert index.header.gameId == "USR3SJiv"
        assert sorted(index.tick_lines) == [tick.tick for tick in ticks]

    def test_read_tick_matches_full_load(self) -> None:
        index = loader.index_snapshots("USR3SJiv")
        _, ticks = loader.load_snapshots("USR3SJiv")

        with paths.snapshot_path("USR3SJiv").open("rb", buffering=0) as file:
            for expected in (ticks[0], ticks[len(ticks) // 2], ticks[-1]):
                offset, length = index.tick_lines[expected.tick]
                assert loader.read_tick(file, offset, length) == expected

    def test_falls_back_to_full_parse_for_unexpected_key_order(self) -> None:
        """The fast path reads the tick number from the line prefix; a line
        with a different key order must still be indexed correctly."""
        header_line, first_tick = FIXTURE.read_text(encoding="utf-8").splitlines()[:2]
        reordered = first_tick.replace('{"kind":"tick","tick":10,', '{"tick":10,"kind":"tick",', 1)
        assert reordered != first_tick
        paths.snapshot_path("USR3SJiv").write_text(
            f"{header_line}\n{reordered}\n", encoding="utf-8"
        )

        index = loader.index_snapshots("USR3SJiv")

        assert list(index.tick_lines) == [10]

    def test_rejects_file_without_header(self) -> None:
        tick_line = FIXTURE.read_text(encoding="utf-8").splitlines()[1]
        paths.snapshot_path("USR3SJiv").write_text(f"{tick_line}\n", encoding="utf-8")

        with pytest.raises(ValueError, match="header"):
            loader.index_snapshots("USR3SJiv")

    def test_read_tick_detects_short_read(self) -> None:
        index = loader.index_snapshots("USR3SJiv")
        offset, length = index.tick_lines[max(index.tick_lines)]

        with paths.snapshot_path("USR3SJiv").open("rb", buffering=0) as file:
            with pytest.raises(ValueError, match="short read"):
                loader.read_tick(file, offset, length + 10)

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="fork-only behavior")
    def test_read_tick_is_safe_on_a_file_shared_across_fork(self) -> None:
        """Forked DataLoader workers inherit one file object (and one OS file
        position); pread-based reads must not disturb each other."""
        index = loader.index_snapshots("USR3SJiv")
        ticks = sorted(index.tick_lines)
        with paths.snapshot_path("USR3SJiv").open("rb", buffering=0) as file:
            pid = os.fork()
            if pid == 0:  # child: hammer the shared file from the other end
                try:
                    for tick in reversed(ticks * 3):
                        loader.read_tick(file, *index.tick_lines[tick])
                finally:
                    os._exit(0)
            for tick in ticks * 3:
                assert loader.read_tick(file, *index.tick_lines[tick]).tick == tick
            _, status = os.waitpid(pid, 0)
            assert status == 0
