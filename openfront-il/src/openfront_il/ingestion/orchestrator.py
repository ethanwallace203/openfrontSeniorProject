"""Orchestrates the Node fetch step: invokes it as a subprocess, retries
the invocation itself on nonzero exit, and validates the raw files it was
supposed to produce.

This module never re-implements HTTP fetching, rate limiting, or
retry-on-network-error logic — all of that lives in
``engine/bridge/src/fetchGame.ts``. It only retries the *subprocess
invocation* if it exits nonzero, and only validates files after the fact.
"""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from openfront_il.ingestion.api_contract import GameRecord
from openfront_il.utils import paths
from openfront_il.utils.logging import get_logger

logger = get_logger(__name__)

FetchStatus = Literal["validated", "missing", "invalid"]


@dataclass(frozen=True)
class FetchResult:
    """The outcome of validating one game's raw file."""

    game_id: str
    status: FetchStatus
    detail: str | None = None


def run_node_fetch(game_ids: Sequence[str], *, max_attempts: int = 3) -> None:
    """Invokes ``npm run fetch -- <gameIds>`` as a subprocess, retrying the
    whole invocation up to ``max_attempts`` times if the process itself
    exits nonzero (e.g. a transient npm/tsx startup failure — HTTP-level
    retries already happen inside fetchGame.ts).

    Args:
        game_ids: gameIds to pass through to the Node CLI.
        max_attempts: how many times to retry the subprocess invocation.

    Raises:
        RuntimeError: if every attempt exits nonzero.
    """
    if not game_ids:
        return
    command = ["npm", "run", "fetch", "--", *game_ids]
    on_windows = platform.system() == "Windows"

    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(
                command,
                cwd=paths.PROJECT_ROOT,
                check=True,
                shell=on_windows,  # npm is a .cmd on Windows and needs shell resolution
            )
            return
        except subprocess.CalledProcessError as error:
            last_error = error
            logger.warning(
                "node fetch invocation failed (attempt %d/%d): %s", attempt, max_attempts, error
            )
    raise RuntimeError(f"node fetch failed after {max_attempts} attempts") from last_error


def validate_raw_files(game_ids: Sequence[str]) -> list[FetchResult]:
    """Checks each expected ``data/raw/{gameId}.json`` exists and validates
    against ``GameRecord``. Never fetches anything itself.
    """
    results: list[FetchResult] = []
    for game_id in game_ids:
        file_path = paths.raw_game_path(game_id)
        if not file_path.exists():
            results.append(FetchResult(game_id, "missing"))
            continue
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
            GameRecord.model_validate(raw)
        except (json.JSONDecodeError, ValidationError) as error:
            results.append(FetchResult(game_id, "invalid", str(error)))
            continue
        results.append(FetchResult(game_id, "validated"))
    return results


def orchestrate_fetch(game_ids: Sequence[str], *, max_attempts: int = 3) -> list[FetchResult]:
    """Fetches whichever of ``game_ids`` aren't already in ``data/raw/``,
    then validates all of them (old and newly fetched).
    """
    missing_ids = [game_id for game_id in game_ids if not paths.raw_game_path(game_id).exists()]
    if missing_ids:
        run_node_fetch(missing_ids, max_attempts=max_attempts)
    return validate_raw_files(game_ids)
