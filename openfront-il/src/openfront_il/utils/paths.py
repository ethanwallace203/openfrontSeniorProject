"""Single source of truth for ``data/``'s on-disk layout.

Mirrors the path constants in ``engine/bridge/src/types.ts`` on the Node
side — the two are kept in sync by hand, and both read the same
``OPENFRONT_DATA_DIR`` environment variable override.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_data_dir() -> Path:
    override = os.environ.get("OPENFRONT_DATA_DIR")
    if override is None:
        return PROJECT_ROOT / "data"
    override_path = Path(override)
    return (
        override_path if override_path.is_absolute() else (PROJECT_ROOT / override_path).resolve()
    )


DATA_DIR = _resolve_data_dir()
RAW_DIR = DATA_DIR / "raw"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
PROCESSED_DIR = DATA_DIR / "processed"


def raw_game_path(game_id: str) -> Path:
    """Path to a game's raw, verbatim API response."""
    return RAW_DIR / f"{game_id}.json"


def snapshot_path(game_id: str) -> Path:
    """Path to a game's per-tick snapshot JSONL file."""
    return SNAPSHOTS_DIR / f"{game_id}.jsonl"
