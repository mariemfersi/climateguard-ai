"""
Event-level train/test split for frequency-severity modeling.

The split unit is YEAR (hurricane season), not individual rows.

This prevents leakage because all locations affected by the same hurricane
season remain together in either train or test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ml.frequency_severity.build_training_table import FEATURE_COLUMNS


def stratified_year_split(
    df: pd.DataFrame,
    year_col: str = "year",
    test_size: float = 0.2,
    seed: int = 42,
    n_bins: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if not 0 < test_size < 1:
        raise ValueError(
            f"test_size must be between 0 and 1. Got {test_size}"
        )

    required_columns = {
        year_col,
        "had_claim"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns for split: {missing}"
        )


    years = df[year_col].unique()

    if len(years) < 2:
        raise ValueError(
            "At least two different years are required."
        )


    # ----------------------------------------------------
    # Compute claim activity per hurricane season
    # ----------------------------------------------------

    year_activity = (
        df.groupby(year_col)["had_claim"]
        .sum()
        .sort_values()
    )


    # Adapt bins if dataset is small
    n_bins = min(
        n_bins,
        len(year_activity)
    )


    year_bins = pd.qcut(
        year_activity,
        q=n_bins,
        labels=False,
        duplicates="drop"
    )


    rng = np.random.default_rng(seed)

    test_years = []


    # ----------------------------------------------------
    # Sample years inside each activity bucket
    # ----------------------------------------------------

    for bin_id in sorted(year_bins.dropna().unique()):

        candidate_years = (
            year_bins[
                year_bins == bin_id
            ]
            .index
            .to_numpy()
        )

        n_test = max(
            1,
            int(round(len(candidate_years) * test_size))
        )


        selected = rng.choice(
            candidate_years,
            size=n_test,
            replace=False
        )

        test_years.extend(selected)


    test_years = set(test_years)

    train_years = (
        set(years)
        -
        test_years
    )


    train_df = (
        df[
            df[year_col]
            .isin(train_years)
        ]
        .reset_index(drop=True)
    )


    test_df = (
        df[
            df[year_col]
            .isin(test_years)
        ]
        .reset_index(drop=True)
    )


    return train_df, test_df
    


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing FEATURE_COLUMNS with sensible defaults so downstream
    code can index FEATURE_COLUMNS without KeyError (useful for synthetic
    test fixtures that omit optional columns).
    """
    df = df.copy()

    for c in FEATURE_COLUMNS:
        if c == "location_id":
            continue
        if c not in df.columns:
            # Categorical-ish columns
            if c in {"metro_center", "construction_class", "roof_type"}:
                df[c] = ""
            else:
                df[c] = 0.0

    return df



def assert_no_year_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    year_col: str = "year",
):

    overlap = (
        set(train_df[year_col])
        &
        set(test_df[year_col])
    )

    if overlap:
        raise AssertionError(
            f"Year leakage detected: {overlap}"
        )

    return True


# Backwards-compatible API: tests and older code expect
# `event_level_train_test_split` to exist. Provide a thin wrapper
# around `stratified_year_split` for clarity and compatibility.
def event_level_train_test_split(
    df: pd.DataFrame,
    year_col: str = "year",
    test_size: float = 0.2,
    seed: int = 42,
    n_bins: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alias preserving older API name: performs an event-level
    (year/season) stratified train/test split.
    """
    return stratified_year_split(
        df, year_col=year_col, test_size=test_size, seed=seed, n_bins=n_bins
    )


def event_level_train_test_split_with_features(
    df: pd.DataFrame,
    year_col: str = "year",
    test_size: float = 0.2,
    seed: int = 42,
    n_bins: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper: performs the stratified split and ensures
    returned DataFrames contain all FEATURE_COLUMNS (with defaults).
    Older tests call `event_level_train_test_split`; keep that API by
    monkey-patching the name to this function below if desired.
    """
    train_df, test_df = stratified_year_split(
        df, year_col=year_col, test_size=test_size, seed=seed, n_bins=n_bins
    )

    train_df = _ensure_feature_columns(train_df)
    test_df = _ensure_feature_columns(test_df)

    return train_df, test_df


# Keep the original legacy name pointing to the compatibility wrapper so
# existing callers and tests continue to work.
event_level_train_test_split = event_level_train_test_split_with_features