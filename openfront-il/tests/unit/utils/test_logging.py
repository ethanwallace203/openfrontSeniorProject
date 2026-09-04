import logging

import pytest

from openfront_il.utils.logging import get_logger


def test_get_logger_does_not_double_prefix_already_qualified_names() -> None:
    # Callers pass __name__, which is already fully-qualified (e.g.
    # "openfront_il.features.dataset_builder") — get_logger must not
    # prepend "openfront_il." again on top of that.
    logger = get_logger("openfront_il.some.module")
    assert logger.name == "openfront_il.some.module"


def test_child_logger_propagates_to_the_configured_root(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_logger("openfront_il.some.other.module")
    caplog.set_level(logging.INFO, logger="openfront_il")
    logger.info("propagated")
    assert any(record.message == "propagated" for record in caplog.records)


def test_get_logger_returns_a_working_logger(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_logger(__name__)
    caplog.set_level(logging.INFO, logger=logger.name)
    logger.info("hello")
    assert any(record.message == "hello" for record in caplog.records)
