"""Process-level tests for ``DaemonClient``: a scripted fake daemon for the
failure modes that are hard to trigger on demand (crash, hang, a stderr
flood), and the real ``ilDaemon.ts`` end to end — the phase-2 done
criterion: live ticks go through the unchanged training featurizer."""

import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from openfront_il.env.daemon_client import (
    DaemonClient,
    DaemonCrashedError,
    DaemonError,
    StepReply,
    default_command,
)
from openfront_il.features.observation_encoder import (
    _IS_LAND_BIT,
    _PLAYER_ID_MASK,
    decode_terrain,
    decode_tile_state,
    encode_observation,
    encode_player_candidates,
)

pytestmark = pytest.mark.slow

FAKE_DAEMON = textwrap.dedent(
    """
    import json, sys
    mode = sys.argv[1]
    for line in sys.stdin:
        op = json.loads(line)["op"]
        if op == "shutdown":
            print(json.dumps({"ok": True}), flush=True)
            break
        if mode == "refuse":
            print(json.dumps({"error": "unknown map Nowhere"}), flush=True)
        elif mode == "crash":
            sys.stderr.write("boom: engine exploded\\n")
            sys.exit(3)
        elif mode == "hang":
            pass
        elif mode == "noisy":
            sys.stderr.write("x" * 256 * 1024 + "\\n")
            sys.stderr.flush()
            print(json.dumps({"error": "still answering"}), flush=True)
    """
)


@pytest.fixture
def fake_daemon(tmp_path: Path) -> Callable[[str], list[str]]:
    script = tmp_path / "fake_daemon.py"
    script.write_text(FAKE_DAEMON, encoding="utf-8")

    def command(mode: str) -> list[str]:
        return [sys.executable, str(script), mode]

    return command


def test_refused_request_keeps_the_client_usable(fake_daemon: Callable[[str], list[str]]) -> None:
    client = DaemonClient(fake_daemon("refuse"))
    with pytest.raises(DaemonError, match="unknown map"):
        client.reset(map_name="Nowhere")
    with pytest.raises(DaemonError):
        client.step()

    client.close()
    # A clean exit means the shutdown request was still answered in sync.
    assert client._process.returncode == 0


def test_crash_reports_exit_code_and_stderr_then_refuses_further_use(
    fake_daemon: Callable[[str], list[str]],
) -> None:
    with DaemonClient(fake_daemon("crash")) as client:
        with pytest.raises(DaemonCrashedError, match="code 3") as crashed:
            client.reset()
        assert "boom: engine exploded" in str(crashed.value)

        with pytest.raises(DaemonCrashedError, match="closed or broken"):
            client.step()


def test_hung_daemon_times_out_and_is_killed(fake_daemon: Callable[[str], list[str]]) -> None:
    with DaemonClient(fake_daemon("hang"), timeout=0.5) as client:
        with pytest.raises(DaemonCrashedError, match="no reply"):
            client.reset()
        assert client._process.poll() is not None


def test_stderr_flood_does_not_deadlock(fake_daemon: Callable[[str], list[str]]) -> None:
    # 256KB is well past a pipe buffer; without the stderr pump the fake
    # would block mid-write and this would time out instead.
    with DaemonClient(fake_daemon("noisy"), timeout=10) as client:
        with pytest.raises(DaemonError, match="still answering"):
            client.reset()


def _spawn_tile(width: int, terrain: np.ndarray, owners: np.ndarray) -> int:
    """First unowned land tile past the map's midpoint whose 4 neighbors
    are land too — read with the encoder's own bit layout."""
    is_land = ((terrain >> _IS_LAND_BIT) & 1).astype(bool)
    free = is_land & (owners == 0)
    height = len(terrain) // width
    for ref in range(len(terrain) // 2, len(terrain)):
        x, y = ref % width, ref // width
        if not free[ref] or not (0 < x < width - 1 and 0 < y < height - 1):
            continue
        if all(is_land[n] for n in (ref - 1, ref + 1, ref - width, ref + width)):
            return ref
    raise AssertionError("no spawn tile found")


@pytest.mark.skipif(
    not Path(default_command()[0]).exists(), reason="node_modules not installed (npm install)"
)
def test_real_daemon_ticks_go_through_the_training_featurizer() -> None:
    with DaemonClient() as daemon:
        first = daemon.reset(seed="py-integration", bots=4)
        header, me = first.header, first.me
        assert first.tick.spawnPhase

        owners = decode_tile_state(header, first.tick).ravel() & _PLAYER_ID_MASK
        tile = _spawn_tile(header.width, decode_terrain(header).ravel(), owners)
        reply: StepReply = daemon.step([{"type": "spawn", "tile": tile}])
        assert reply.rejected == []

        spatial_shape = flat_shape = None
        previous_tick = reply.tick.tick
        for _ in range(50):
            reply = daemon.step()
            assert reply.tick.tick == previous_tick + header.snapshotEvery
            previous_tick = reply.tick.tick

            spatial, flat = encode_observation(header, reply.tick, me)
            small_ids, player_features = encode_player_candidates(header, reply.tick, me)
            assert np.isfinite(spatial).all() and np.isfinite(flat).all()
            assert np.isfinite(player_features).all()
            assert len(small_ids) == len(player_features)
            spatial_shape = spatial_shape or spatial.shape
            flat_shape = flat_shape or flat.shape
            assert spatial.shape == spatial_shape and flat.shape == flat_shape

        assert not reply.tick.spawnPhase
        assert reply.alive
