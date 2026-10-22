"""Python side of ``engine/bridge/src/ilDaemon.ts``: starts the daemon as a
subprocess, sends it reset/step requests, and returns the replies as the
same ``SnapshotHeader``/``TickSnapshot`` models the training pipeline reads
from disk.

This is the one place outside ``ingestion/orchestrator.py`` allowed to run
engine code, and like the orchestrator it only ever does so by invoking the
Node CLI as a subprocess — it never imports engine code. See
``src/openfront_il/env/README.md`` for why live play needs a running process
where the offline pipeline uses files.

The protocol (one JSON request per stdin line, exactly one JSON reply per
stdout line) is documented in ``ilDaemon.ts``'s header, which is its source
of truth; the envelope models below mirror it with ``extra="forbid"`` so a
protocol change on the TypeScript side fails the first reply loudly.
"""

from __future__ import annotations

import collections
import json
import platform
import queue
import subprocess
import threading
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from openfront_il.replay.snapshot_schema import SnapshotHeader, TickSnapshot
from openfront_il.utils import paths
from openfront_il.utils.logging import get_logger

logger = get_logger(__name__)

# An engine intent as sent to the daemon, e.g. {"type": "spawn", "tile": 123}.
# Its internals are features/action_decoder.py's concern, not this module's.
Intent = dict[str, object]

_STDERR_TAIL_LINES = 200


class DaemonError(RuntimeError):
    """The daemon replied ``{"error": ...}`` to a request. The process is
    still alive and the client can keep being used."""


class DaemonCrashedError(RuntimeError):
    """The daemon process exited, timed out or broke the protocol. The
    client is no longer usable; the message carries the daemon's last
    stderr lines."""


# ---------------------------------------------------------------------------
# Reply envelopes (mirror ilDaemon.ts's ResetReply / StepReply)
# ---------------------------------------------------------------------------


class RejectedIntent(BaseModel):
    """An intent the daemon refused before it reached the engine."""

    model_config = ConfigDict(extra="forbid")

    index: int
    intent: object
    reason: str


class ResetReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    header: SnapshotHeader
    tick: TickSnapshot
    me: int


class StepReply(BaseModel):
    """``alive`` is false both before the agent spawns and after it dies;
    ``tick.spawnPhase`` tells the two apart."""

    model_config = ConfigDict(extra="forbid")

    tick: TickSnapshot
    me: int
    alive: bool
    rejected: list[RejectedIntent]


class _OkReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool


class _ErrorReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str


ReplyT = TypeVar("ReplyT", bound=BaseModel)


def parse_reply(line: str | bytes, expected: type[ReplyT]) -> ReplyT:
    """Parses one daemon stdout line into ``expected``.

    Pure function (no process involved), so the protocol handling is unit
    testable on its own. The common case is a single ``model_validate_json``
    pass; the line is only decoded again to diagnose a failure.

    Raises:
        DaemonError: the line is an ``{"error": ...}`` reply.
        DaemonCrashedError: the line is neither valid JSON nor the expected
            shape — the two sides disagree about the protocol.
    """
    try:
        return expected.model_validate_json(line)
    except ValidationError as validation_error:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            raise DaemonCrashedError(f"daemon wrote a non-JSON line: {line!r:.200}") from None
        if isinstance(raw, dict) and set(raw) == {"error"}:
            raise DaemonError(_ErrorReply.model_validate(raw).error) from None
        raise DaemonCrashedError(
            f"daemon reply doesn't match {expected.__name__} — has ilDaemon.ts's "
            f"protocol changed without daemon_client.py? {validation_error}"
        ) from validation_error


# ---------------------------------------------------------------------------
# Process wrapper
# ---------------------------------------------------------------------------


def default_command() -> list[str]:
    """``tsx`` invoked directly rather than via ``npm run daemon``: npm
    prints a banner to stdout (corrupting the first reply) and sits between
    us and the daemon as an extra process that may not forward signals."""
    tsx = "tsx.cmd" if platform.system() == "Windows" else "tsx"
    return [
        str(paths.PROJECT_ROOT / "node_modules" / ".bin" / tsx),
        "--tsconfig",
        "tsconfig.run.json",
        "engine/bridge/src/ilDaemon.ts",
    ]


class DaemonClient:
    """One live game, driven over the daemon's stdio protocol.

    Use as a context manager so the process is always shut down::

        with DaemonClient() as daemon:
            first = daemon.reset(map_name="Onion", seed="0")
            reply = daemon.step([{"type": "spawn", "tile": 1234}])

    Raises ``DaemonError`` for requests the daemon refused (the client
    stays usable) and ``DaemonCrashedError`` once the process is gone, hung or
    out of protocol sync (the client is then unusable).
    """

    def __init__(self, command: Sequence[str] | None = None, *, timeout: float = 120.0) -> None:
        self._timeout = timeout
        self._broken = False
        self._closed = False
        self._process = subprocess.Popen(
            list(command) if command is not None else default_command(),
            cwd=paths.PROJECT_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            shell=platform.system() == "Windows",  # tsx is a .cmd on Windows
        )
        logger.debug("started il daemon (pid %d)", self._process.pid)

        # Lines from stdout; None marks end of stream (the process exited).
        self._replies: queue.Queue[str | None] = queue.Queue()
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=_STDERR_TAIL_LINES)
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        # stderr must be drained continuously: an unread pipe fills (~64KB)
        # and the daemon then blocks mid-write, deadlocking both sides.
        self._stderr_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_thread.start()

    # -- public API --------------------------------------------------------

    def reset(
        self,
        *,
        map_name: str | None = None,
        seed: str | None = None,
        bots: int | None = None,
        difficulty: str | None = None,
        nations: int | str | None = None,
        game_type: str | None = None,
        snapshot_every: int | None = None,
    ) -> ResetReply:
        """Boots a fresh game. Only options that are set are sent — the
        defaults live in one place, ``ilDaemon.ts``."""
        options = {
            "map": map_name,
            "seed": seed,
            "bots": bots,
            "difficulty": difficulty,
            "nations": nations,
            "gameType": game_type,
            "snapshotEvery": snapshot_every,
        }
        payload = {"op": "reset", **{k: v for k, v in options.items() if v is not None}}
        return self._request(payload, ResetReply)

    def step(self, intents: Sequence[Intent] = (), ticks: int | None = None) -> StepReply:
        """Sends ``intents`` as one turn and advances ``ticks`` ticks (the
        reset's ``snapshot_every`` when omitted)."""
        payload: dict[str, object] = {"op": "step", "intents": list(intents)}
        if ticks is not None:
            payload["ticks"] = ticks
        return self._request(payload, StepReply)

    def close(self, timeout: float = 10.0) -> None:
        """Shuts the daemon down: a polite ``shutdown`` request if the
        protocol is still in sync, then terminate, then kill. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if not self._broken and self._process.poll() is None:
            try:
                self._request({"op": "shutdown"}, _OkReply, allow_closed=True)
            except (DaemonError, DaemonCrashedError):
                pass
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()

    def __enter__(self) -> DaemonClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _request(
        self, payload: Mapping[str, object], expected: type[ReplyT], *, allow_closed: bool = False
    ) -> ReplyT:
        if self._broken or (self._closed and not allow_closed):
            raise DaemonCrashedError("daemon client is closed or broken; create a new one")
        stdin = self._process.stdin
        if stdin is None:
            raise self._crash("daemon has no stdin")
        try:
            stdin.write(json.dumps(payload) + "\n")
            stdin.flush()
        except OSError as error:
            raise self._crash(f"could not write to daemon: {error}") from error

        try:
            line = self._replies.get(timeout=self._timeout)
        except queue.Empty:
            raise self._crash(f"no reply to {payload['op']!r} within {self._timeout}s") from None
        if line is None:
            raise self._crash(f"daemon exited (code {self._process.poll()})")

        try:
            return parse_reply(line, expected)
        except DaemonCrashedError as error:
            raise self._crash(str(error)) from error

    def _crash(self, reason: str) -> DaemonCrashedError:
        """Marks the client unusable, kills the process, and builds the
        error — after giving the stderr pump a moment to collect the
        daemon's last words (usually the real stack trace)."""
        self._broken = True
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        self._stderr_thread.join(timeout=1.0)
        tail = "\n".join(self._stderr_tail) or "(no stderr output)"
        return DaemonCrashedError(f"{reason}\n--- daemon stderr (last lines) ---\n{tail}")

    def _pump_stdout(self) -> None:
        stdout = self._process.stdout
        if stdout is not None:
            for line in stdout:
                self._replies.put(line)
        self._replies.put(None)

    def _pump_stderr(self) -> None:
        stderr = self._process.stderr
        if stderr is None:
            return
        for line in stderr:
            stripped = line.rstrip("\n")
            self._stderr_tail.append(stripped)
            logger.debug("daemon: %s", stripped)
