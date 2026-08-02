"""
Aggregate the Phase 2 simulated claims table into per-location historical
hazard summary features.

DESIGN NOTE: this reuses claims.parquet (Milestone 2.2) rather than
recomputing storm-distance-decay from scratch. claims.parquet already IS
the distance-decayed wind exposure per (location, historical storm) —
recomputing an equivalent feature here would be redundant and could
silently drift out of sync with the claims-generation methodology.
"""

from __future__ import annotations

import pandas as pd


def aggregate_hazard_features_per_location(
    locations: pd.DataFrame, claims: pd.DataFrame
) -> pd.DataFrame:
    """
    Args:
        locations: must have 'location_id' (defines the full universe —
            locations with ZERO historical claims must still appear, with
            zeroed-out hazard features, not be silently dropped).
        claims: output of claims_generator.generate_claims().

    Returns:
        DataFrame with one row per location_id:
        [location_id, historical_claim_count, historical_incurred_loss_usd,
        historical_max_wind_kt, historical_mean_damage_ratio,
        distinct_storms_experienced].
    """
    agg = (
        claims.groupby("location_id")
        .agg(
            historical_claim_count=("claim_id", "count"),
            historical_incurred_loss_usd=("incurred_loss_usd", "sum"),
            historical_max_wind_kt=("max_wind_experienced_kt", "max"),
            historical_mean_damage_ratio=("damage_ratio", "mean"),
            distinct_storms_experienced=("storm_id", "nunique"),
        )
        .reset_index()
    )

    # Left-join against the FULL location universe so locations with no
    # historical claims are retained with zeroed hazard features, rather
    # than silently disappearing from the Gold table.
    result = locations[["location_id"]].merge(agg, on="location_id", how="left")
    fill_cols = [
        "historical_claim_count",
        "historical_incurred_loss_usd",
        "historical_max_wind_kt",
        "historical_mean_damage_ratio",
        "distinct_storms_experienced",
    ]
    result[fill_cols] = result[fill_cols].fillna(0)
    result["historical_claim_count"] = result["historical_claim_count"].astype(int)
    result["distinct_storms_experienced"] = result["distinct_storms_experienced"].astype(int)

    return result