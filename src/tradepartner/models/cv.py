"""Purged, embargoed walk-forward splits (spec 10.5).

Expanding train window up to T, then an embargo equal to the label horizon, then a test window
of ``test_months``, rolling forward. A training row at position ``i`` has a label window that
ends at ``i + horizon``; it is only kept if that is strictly before the test window starts, so no
label ever overlaps the test period (purge + embargo in one rule).
"""

from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass(frozen=True)
class Split:
    window: int
    train_idx: np.ndarray  # positions into the sorted date list
    test_idx: np.ndarray
    train_end: date
    test_start: date
    test_end: date


def _month_key(d: date) -> int:
    return d.year * 12 + d.month - 1


def walk_forward_splits(
    dates: list[date],
    horizon: int,
    test_months: int = 3,
    min_train_days: int = 504,
) -> list[Split]:
    """``dates`` are the sorted distinct trading dates of the matrix."""
    if dates != sorted(dates):
        raise ValueError("dates must be sorted")
    if not dates:
        return []
    months = [_month_key(d) for d in dates]
    out: list[Split] = []
    first_month = months[0]
    # test blocks start on month boundaries that leave enough history for training
    block_start_month = first_month
    while True:
        in_block = [
            i
            for i, m in enumerate(months)
            if block_start_month <= m < block_start_month + test_months
        ]
        if not in_block:
            if block_start_month > months[-1]:
                break
            block_start_month += test_months
            continue
        ts = in_block[0]
        last_train = ts - horizon - 1  # embargo = horizon: label window ends before the test
        if last_train + 1 >= min_train_days:
            train = np.arange(0, last_train + 1)
            out.append(
                Split(
                    len(out),
                    train,
                    np.array(in_block),
                    dates[last_train],
                    dates[ts],
                    dates[in_block[-1]],
                )
            )
        block_start_month += test_months
    return out


def assert_no_overlap(split: Split, horizon: int) -> None:
    """Raise if any training label window reaches into the test period (used in tests/CI)."""
    if len(split.train_idx) and split.train_idx.max() + horizon >= split.test_idx.min():
        raise AssertionError("training label window overlaps the test period")


def inner_split(n: int, horizon: int, valid_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Time-ordered train/validation split inside a training fold, with the same embargo."""
    cut = int(n * (1 - valid_fraction))
    return np.arange(0, max(0, cut - horizon - 1)), np.arange(cut, n)
