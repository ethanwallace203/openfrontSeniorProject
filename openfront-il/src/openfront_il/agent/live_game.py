"""Python handle on one live, headless OpenFront game.

Starts ``engine/bridge/src/liveGame.ts`` as a long-lived Node subprocess
and talks to it over stdio, one JSON object per line (protocol documented
at the top of that file). Observations come back as the same
``SnapshotHeader`` / ``TickSnapshot`` pydantic models the training data
uses, so a policy sees exactly the format it was trained on.

Like ``ingestion/orchestrator.py``, this never imports engine code — it
only invokes the Node CLI as a subprocess.

Usage::

    with LiveGame() as game:
        header, tick = game.reset(map="Onion", seed="s1", bots=4)
        while not game.is_done:
            tick = game.step([{"type": "attack", "targetID": None, "troops": None}])
        game.save()
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from openfront_il.replay.snapshot_schema import PlayerScalars, SnapshotHeader, TickSnapshot
from openfront_il.utils import paths

# Must match AGENT_CLIENT_ID in engine/bridge/src/liveGame.ts.
AGENT_CLIENT_ID = "AGENTIL1"

_COMMAND = ["npx", "tsx", "--tsconfig", "tsconfig.run.json", "engine/bridge/src/liveGame.ts"]
_CLOSE_TIMEOUT_SECONDS = 10


class LiveGameError(RuntimeError):
    """The Node side reported an error, or the process died."""


class LiveGame:
    """One Node engine process. Reusable: call ``reset`` again to start a
    new game in the same process (avoids paying Node startup per game).

    Args:
        log_path: where the engine's stderr (its log output) goes. ``None``
            discards it. It's never piped back to Python, because an unread
            pipe fills up and deadlocks the engine.
        project_root: the openfront-il directory (Node runs from here so
            ``tsconfig.run.json`` and relative paths resolve).
    """

    def __init__(self, *, log_path: Path | None = None, project_root: Path = paths.PROJECT_ROOT):
        self._log_file: IO[str] | None = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115

        self._process = subprocess.Popen(
            _COMMAND,
            cwd=project_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_file if self._log_file is not None else subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            shell=platform.system() == "Windows",  # npx is a .cmd on Windows
        )
        self._header: SnapshotHeader | None = None
        self._tick: TickSnapshot | None = None

    # ------------------------------------------------------------------ protocol

    def _send(self, message: dict[str, Any]) -> None:
        if self._process.poll() is not None:
            raise LiveGameError(f"engine process exited with code {self._process.returncode}")
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()

    def _recv(self) -> dict[str, Any]:
        assert self._process.stdout is not None
        line = self._process.stdout.readline()
        if not line:
            code = self._process.wait()
            raise LiveGameError(f"engine process exited unexpectedly with code {code}")
        message: dict[str, Any] = json.loads(line)
        if message.get("kind") == "error":
            raise LiveGameError(message.get("message", "unknown engine error"))
        return message

    # ------------------------------------------------------------------ commands

    def reset(
        self,
        *,
        map: str = "Onion",  # noqa: A002 - matches the engine's own field name
        seed: str | None = None,
        bots: int = 0,
        difficulty: str = "Medium",
        nations: int | str | None = None,
        snapshot_every: int = 10,
    ) -> tuple[SnapshotHeader, TickSnapshot]:
        """Starts a new game and returns its header plus the first
        post-spawn-phase snapshot. ``seed`` makes the game reproducible;
        ``None`` lets the engine pick one from the clock."""
        message: dict[str, Any] = {
            "op": "reset",
            "map": map,
            "bots": bots,
            "difficulty": difficulty,
            "snapshotEvery": snapshot_every,
        }
        if seed is not None:
            message["seed"] = seed
        if nations is not None:
            message["nations"] = nations

        self._send(message)
        self._header = SnapshotHeader.model_validate(self._recv())
        self._tick = TickSnapshot.model_validate(self._recv())
        return self._header, self._tick

    def step(
        self, intents: list[dict[str, Any]] | None = None, ticks: int | None = None
    ) -> TickSnapshot:
        """Submits the agent's intents (may be empty), advances ``ticks``
        game ticks (default: the header's ``snapshotEvery``), and returns
        the new snapshot. Stops early if someone wins. The engine stamps
        ``clientID`` on each intent, so callers don't need to."""
        if self._header is None:
            raise LiveGameError("step() called before reset()")
        message: dict[str, Any] = {"op": "step", "intents": intents or []}
        if ticks is not None:
            message["ticks"] = ticks
        self._send(message)
        self._tick = TickSnapshot.model_validate(self._recv())
        return self._tick

    def save(self, path: Path | None = None) -> Path:
        """Writes the game as a GameRecord JSON (archive format). Defaults
        to ``data/agent_games/{gameId}.json``."""
        if self._header is None:
            raise LiveGameError("save() called before reset()")
        out = (path or paths.AGENT_GAMES_DIR / f"{self._header.gameId}.json").resolve()
        self._send({"op": "save", "path": str(out)})
        self._recv()
        return out

    def close(self) -> None:
        """Shuts the engine down. Safe to call more than once."""
        if self._process.poll() is None:
            try:
                self._send({"op": "close"})
                self._process.wait(timeout=_CLOSE_TIMEOUT_SECONDS)
            except (LiveGameError, OSError, subprocess.TimeoutExpired):
                self._process.kill()
                self._process.wait()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> LiveGame:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ state

    @property
    def header(self) -> SnapshotHeader:
        if self._header is None:
            raise LiveGameError("no game running - call reset() first")
        return self._header

    @property
    def tick(self) -> TickSnapshot:
        if self._tick is None:
            raise LiveGameError("no game running - call reset() first")
        return self._tick

    @property
    def agent(self) -> PlayerScalars | None:
        """The agent's own player row in the latest snapshot."""
        return next((p for p in self.tick.players if p.clientID == AGENT_CLIENT_ID), None)

    @property
    def agent_alive(self) -> bool:
        # The engine only computes legality for living humans, so this is
        # the same test dataset_builder.py uses for "still playing".
        return AGENT_CLIENT_ID in self.tick.legal

    @property
    def agent_won(self) -> bool:
        # The engine reports a human winner as ["player", clientID]
        # (GameImpl.makeWinner); nations come back as ["nation", name].
        winner = self.tick.winner
        return winner is not None and winner.kind == "player" and winner.id == AGENT_CLIENT_ID

    @property
    def is_done(self) -> bool:
        """True once someone has won or the agent has been eliminated."""
        return self.tick.winner is not None or not self.agent_alive
