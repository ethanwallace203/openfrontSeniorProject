"""Pure metric functions: ``(predictions, labels[, legality_mask]) ->
score``. No I/O, no checkpoint loading — ``held_out_report.py`` owns that.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def accuracy(predictions: NDArray[np.int64], labels: NDArray[np.int64]) -> float:
    """Fraction of ``predictions`` exactly matching ``labels``. ``0.0`` for
    an empty input (not ``NaN``) — an empty held-out set is reported as
    "nothing to measure," not silently propagated as a NaN.
    """
    if len(labels) == 0:
        return 0.0
    return float((predictions == labels).mean())


def most_common_class_baseline_accuracy(labels: NDArray[np.int64], num_classes: int) -> float:
    """Accuracy of always predicting the plurality class in ``labels`` —
    the trivial baseline ``docs/imitation_learning_plan.md`` Section 5
    says the model must beat; raw accuracy alone is not a meaningful
    number on an imbalanced label distribution.
    """
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels, minlength=num_classes)
    return float(counts.max() / len(labels))


def non_excluded_class_accuracy(
    predictions: NDArray[np.int64], labels: NDArray[np.int64], excluded_class: int
) -> float:
    """Accuracy restricted to examples whose true label isn't
    ``excluded_class`` (typically ``noop``) — the metric that actually
    distinguishes "predicts real actions well" from "just predicts noop
    for everything," which raw accuracy on an noop-dominated distribution
    cannot (see ``docs/imitation_learning_plan.md`` Section 5).
    """
    mask = labels != excluded_class
    if not mask.any():
        return 0.0
    return accuracy(predictions[mask], labels[mask])


def legality_violation_rate(
    predictions: NDArray[np.int64], legality_masks: NDArray[np.bool_]
) -> float:
    """Fraction of ``predictions`` that land on a masked-illegal class.

    Only meaningful for action types with a real backing legality signal
    — see ``docs/imitation_learning_plan.md`` Section 2 for which those
    are; a class defaulting to "legal" because no signal was tracked will
    never count as a violation here regardless of whether it was actually
    sensible.
    """
    if len(predictions) == 0:
        return 0.0
    is_illegal = ~legality_masks[np.arange(len(predictions)), predictions]
    return float(is_illegal.mean())
