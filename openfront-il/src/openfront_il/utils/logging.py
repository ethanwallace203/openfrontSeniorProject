"""Structured logging setup, used everywhere in ``src/`` instead of ``print()``.

``scripts/`` may use ``print()`` for CLI-facing output — that exception is
documented in ``scripts/README.md``.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.environ.get("OPENFRONT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
        )
    )
    root = logging.getLogger("openfront_il")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Returns a logger configured with a consistent formatter on first
    use.

    Args:
        name: ``__name__`` of the calling module. Every module under
            ``src/openfront_il/`` already has a fully-qualified dotted
            name starting with ``openfront_il`` (e.g.
            ``openfront_il.features.dataset_builder``) — this does *not*
            prepend anything, since doing so would double up that prefix.
            The ``"openfront_il"`` root logger configured in
            ``_configure_root`` still receives every one of these via
            normal logger-hierarchy propagation.

    Returns:
        A configured ``logging.Logger``.
    """
    _configure_root()
    return logging.getLogger(name)
