from openfront_il.utils import paths


def test_data_dir_is_isolated_test_dir() -> None:
    # tests/conftest.py points OPENFRONT_DATA_DIR at a throwaway temp dir
    # before this module is imported.
    assert "openfront-il-test-" in str(paths.DATA_DIR)


def test_raw_game_path() -> None:
    result = paths.raw_game_path("abc123")
    assert result == paths.RAW_DIR / "abc123.json"
    assert result.name == "abc123.json"


def test_snapshot_path() -> None:
    result = paths.snapshot_path("abc123")
    assert result == paths.SNAPSHOTS_DIR / "abc123.jsonl"
    assert result.name == "abc123.jsonl"


def test_subdirs_nest_under_data_dir() -> None:
    assert paths.RAW_DIR.parent == paths.DATA_DIR
    assert paths.SNAPSHOTS_DIR.parent == paths.DATA_DIR
    assert paths.PROCESSED_DIR.parent == paths.DATA_DIR
