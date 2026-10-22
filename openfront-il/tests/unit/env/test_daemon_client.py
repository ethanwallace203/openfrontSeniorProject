"""Pure protocol tests for ``parse_reply`` — no process involved. The
process-level behavior (timeouts, crashes, the real daemon) lives in
``tests/integration/test_daemon_process.py``."""

import json
from pathlib import Path

import pytest

from openfront_il.env.daemon_client import (
    DaemonCrashedError,
    DaemonError,
    ResetReply,
    StepReply,
    parse_reply,
)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "USR3SJiv.snapshots.jsonl"


def _fixture_lines() -> tuple[dict[str, object], dict[str, object]]:
    with FIXTURE.open(encoding="utf-8") as file:
        header = json.loads(file.readline())
        tick = json.loads(file.readline())
    return header, tick


def _step_reply(**overrides: object) -> dict[str, object]:
    _, tick = _fixture_lines()
    return {"tick": tick, "me": 1, "alive": True, "rejected": [], **overrides}


def test_parses_a_reset_reply_into_snapshot_models() -> None:
    header, tick = _fixture_lines()

    reply = parse_reply(json.dumps({"header": header, "tick": tick, "me": 1}), ResetReply)

    assert reply.header.gameId == "USR3SJiv"
    assert reply.tick.tick == tick["tick"]
    assert reply.me == 1


def test_parses_a_step_reply_with_rejected_intents() -> None:
    rejected = [{"index": 0, "intent": {"type": "spawn"}, "reason": "tile: required"}]

    reply = parse_reply(json.dumps(_step_reply(rejected=rejected)), StepReply)

    assert reply.alive is True
    assert reply.rejected[0].index == 0
    assert reply.rejected[0].reason == "tile: required"


def test_error_reply_raises_daemon_error_with_its_message() -> None:
    with pytest.raises(DaemonError, match="step before reset"):
        parse_reply(json.dumps({"error": "step before reset"}), StepReply)


def test_non_json_line_raises_crashed() -> None:
    with pytest.raises(DaemonCrashedError, match="non-JSON"):
        parse_reply("> openfront-il-engine-bridge@0.1.0 daemon", StepReply)


def test_unexpected_field_raises_crashed() -> None:
    # The protocol-drift alarm: ilDaemon.ts grew a field this side doesn't know.
    with pytest.raises(DaemonCrashedError, match="protocol changed"):
        parse_reply(json.dumps(_step_reply(newField=1)), StepReply)


def test_missing_field_raises_crashed() -> None:
    reply = _step_reply()
    del reply["alive"]

    with pytest.raises(DaemonCrashedError, match="protocol changed"):
        parse_reply(json.dumps(reply), StepReply)
