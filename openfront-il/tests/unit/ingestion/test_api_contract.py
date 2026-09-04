import json
from pathlib import Path

import pytest

from openfront_il.ingestion.api_contract import GameRecord

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.mark.parametrize("game_id", ["USR3SJiv", "vmVhD8B3"])
def test_validates_real_raw_game_records(game_id: str) -> None:
    raw = json.loads((FIXTURES_DIR / f"{game_id}.json").read_text(encoding="utf-8"))
    record = GameRecord.model_validate(raw)
    assert record.info.gameID == game_id
    assert len(record.turns) > 0


def test_rejects_a_record_missing_required_fields() -> None:
    with pytest.raises(Exception):  # noqa: B017 - pydantic's ValidationError
        GameRecord.model_validate({"info": {}})


def test_tolerates_unknown_extra_fields() -> None:
    raw = json.loads((FIXTURES_DIR / "USR3SJiv.json").read_text(encoding="utf-8"))
    raw["someBrandNewFieldTheApiAddedLater"] = 42
    raw["info"]["config"]["someNewConfigOption"] = True
    record = GameRecord.model_validate(raw)
    assert record.info.gameID == "USR3SJiv"


def test_purchased_tribe_names_are_parsed() -> None:
    raw = json.loads((FIXTURES_DIR / "vmVhD8B3.json").read_text(encoding="utf-8"))
    record = GameRecord.model_validate(raw)
    assert record.info.tribes is not None
    assert len(record.info.tribes) == 10
    assert all(isinstance(t.name, str) and t.name for t in record.info.tribes)
