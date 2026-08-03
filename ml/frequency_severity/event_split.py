"""
Event-level train/test split for the frequency-severity training table.

METHODOLOGY NOTE: per the roadmap, event-level (not row-level) splitting is
a non-negotiable requirement to prevent data leakage. Here, "event" =
hurricane season YEAR. A naive row-level random split would let a
location's 1992 row land in train while its 1993 row lands in test — that
alone isn't leakage (a location isn't an "event"). The real leakage risk
this guards against is subtler: many DIFFERENT locations share highly
correlated outcomes within the same year (a single intense season affects
many locations at once). If some of that year's rows leaked into train and
others into test, a model could achieve inflated apparent performance by
partially "memorizing" that year's specific storm activity rather than
genuinely generalizing to unseen seasons. Splitting by year guarantees an
entire season's rows go entirely to one side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def event_level_train_test_split(
    df: pd.DataFrame,
    year_col: str = "year",
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split `df` into train/test by randomly assigning whole YEARS to each
    side — never splitting a single year's rows across both.

    Args:
        df: training table with a year column (e.g. build_training_table's
            output).
        year_col: name of the year column.
        test_size: fraction of DISTINCT YEARS (not rows) held out for test.
        seed: for reproducibility.

    Returns:
        (train_df, test_df)

    Raises:
        ValueError: if test_size is not in (0, 1), or fewer than 2 distinct
            years are present (can't do a meaningful split with 0-1 years).
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    years = df[year_col].unique()
    if len(years) < 2:
        raise ValueError(
            f"Need at least 2 distinct years to split, got {len(years)}"
        )

    rng = np.random.default_rng(seed)
    shuffled_years = rng.permutation(years)
    n_test_years = max(1, round(len(years) * test_size))

    test_years = set(shuffled_years[:n_test_years])
    train_years = set(shuffled_years[n_test_years:])

    train_df = df[df[year_col].isin(train_years)].reset_index(drop=True)
    test_df = df[df[year_col].isin(test_years)].reset_index(drop=True)

    return train_df, test_df