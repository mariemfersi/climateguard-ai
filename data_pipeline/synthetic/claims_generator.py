"""
Claims generator: runs the historical HURDAT2 storm catalog against the
current synthetic book of business, producing simulated claims grounded in
real storm physics (via wind_field_model) and a documented vulnerability
curve (via vulnerability_curve).

METHODOLOGY NOTE — how historical storms are applied to the current book:
Consistent with standard catastrophe-modeling practice, historical storms
are treated as a STOCHASTIC EVENT CATALOG applied to the CURRENT exposure
(today's synthetic book), not constrained to have occurred during any
individual policy's actual historical effective_date window. Real cat
models use historical event sets this same way — as a statistical sample
of possible events, re-applied to present-day exposure — not as a literal
historical replay. This is standard actuarial/cat-modeling methodology,
not a shortcut specific to this project.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data_pipeline.synthetic.vulnerability_curve import damage_ratio
from data_pipeline.synthetic.wind_field_model import (
    estimate_rmw_km,
    haversine_km,
    wind_speed_at_distance,
)

logger = logging.getLogger(__name__)

# Buffered well beyond the FLORIDA_BBOX used for location generation, since
# a storm's wind field extends hundreds of km beyond its track center — a
# storm whose center never enters Florida can still produce damaging wind
# at Florida locations (e.g. a Gulf storm passing 200km offshore).
FLORIDA_INFLUENCE_BBOX = {"lat_min": 18.0, "lat_max": 36.0, "lon_min": -93.0, "lon_max": -74.0}

MIN_WIND_THRESHOLD_KT = 50.0  # below this, damage is treated as negligible (perf + realism)
MIN_DAMAGE_RATIO_THRESHOLD = 0.005  # below this, don't bother emitting a claim row


def filter_storms_near_florida(
    hurdat2_df: pd.DataFrame, bbox: dict = FLORIDA_INFLUENCE_BBOX
) -> list[str]:
    """
    Cheap pre-filter: only storms with at least one track point inside the
    buffered Florida-influence bounding box are worth the expensive
    per-location wind field computation. Drastically reduces compute
    (most Atlantic storms never approach Florida at all).
    """
    mask = hurdat2_df["lat"].between(bbox["lat_min"], bbox["lat_max"]) & hurdat2_df[
        "lon"
    ].between(bbox["lon_min"], bbox["lon_max"])
    return sorted(hurdat2_df.loc[mask, "storm_id"].unique().tolist())


def _compute_wind_and_timing(
    loc_lat: np.ndarray,
    loc_lon: np.ndarray,
    track_lat: np.ndarray,
    track_lon: np.ndarray,
    track_wind_kt: np.ndarray,
    track_timestamps: np.ndarray,
    max_influence_km: float = 400.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For N locations and a single storm's T track points, return both the
    peak experienced wind AND the timestamp at which that peak occurred
    (needed for claims.loss_date) — a superset of
    wind_field_model.max_wind_experienced_by_locations, which only returns
    the peak wind value. Reuses the same tested primitives.
    """
    loc_lat_r = np.asarray(loc_lat, dtype=float).reshape(-1, 1)
    loc_lon_r = np.asarray(loc_lon, dtype=float).reshape(-1, 1)
    track_lat_r = np.asarray(track_lat, dtype=float).reshape(1, -1)
    track_lon_r = np.asarray(track_lon, dtype=float).reshape(1, -1)
    track_wind_r = np.asarray(track_wind_kt, dtype=float).reshape(1, -1)

    distance_km = haversine_km(loc_lat_r, loc_lon_r, track_lat_r, track_lon_r)
    rmw_km = estimate_rmw_km(track_wind_r)
    wind_matrix = wind_speed_at_distance(track_wind_r, distance_km, rmw_km)
    wind_matrix = np.where(distance_km <= max_influence_km, wind_matrix, 0.0)

    max_wind = wind_matrix.max(axis=1)
    argmax_idx = wind_matrix.argmax(axis=1)
    loss_timestamps = np.asarray(track_timestamps)[argmax_idx]
    return max_wind, loss_timestamps


def generate_claims(
    locations: pd.DataFrame,
    policies: pd.DataFrame,
    hurdat2_tracks: pd.DataFrame,
    min_wind_threshold_kt: float = MIN_WIND_THRESHOLD_KT,
    min_damage_ratio_threshold: float = MIN_DAMAGE_RATIO_THRESHOLD,
) -> pd.DataFrame:
    """
    Run the filtered historical storm catalog against the current synthetic
    book, producing a simulated claims table.

    Args:
        locations: must have [location_id, lat, lon, construction_class,
            roof_type, tiv_usd] (output of generate_locations +
            assign_attributes).
        policies: must have [location_id, policy_id, deductible_usd,
            limit_usd] (output of generate_policies).
        hurdat2_tracks: HURDAT2 parsed tracks (output of
            data_pipeline.ingestion.hurdat2.parse_hurdat2), with
            [storm_id, name, timestamp, lat, lon, max_wind_kt].

    Returns:
        DataFrame with columns [claim_id, policy_id, location_id, storm_id,
        storm_name, loss_date, peril_type, max_wind_experienced_kt,
        damage_ratio, incurred_loss_usd, paid_loss_usd].
    """
    required_loc_cols = {"location_id", "lat", "lon", "construction_class", "roof_type", "tiv_usd"}
    missing = required_loc_cols - set(locations.columns)
    if missing:
        raise ValueError(f"locations is missing required columns: {missing}")

    required_policy_cols = {"location_id", "policy_id", "deductible_usd", "limit_usd"}
    missing_policy = required_policy_cols - set(policies.columns)
    if missing_policy:
        raise ValueError(f"policies is missing required columns: {missing_policy}")

    storm_ids = filter_storms_near_florida(hurdat2_tracks)
    logger.info(
        "Filtered to %d storms with track points near Florida (out of %d total storms)",
        len(storm_ids),
        hurdat2_tracks["storm_id"].nunique(),
    )

    loc_lat = locations["lat"].to_numpy()
    loc_lon = locations["lon"].to_numpy()
    construction_class = locations["construction_class"].to_numpy()
    roof_type = locations["roof_type"].to_numpy()
    tiv_usd = locations["tiv_usd"].to_numpy()
    location_ids = locations["location_id"].to_numpy()

    policy_lookup = policies[["location_id", "policy_id", "deductible_usd", "limit_usd"]]

    result_frames = []

    for i, storm_id in enumerate(storm_ids):
        storm_track = hurdat2_tracks[hurdat2_tracks["storm_id"] == storm_id].sort_values(
            "timestamp"
        )
        storm_name = storm_track["name"].iloc[0]

        max_wind, loss_timestamps = _compute_wind_and_timing(
            loc_lat,
            loc_lon,
            storm_track["lat"].to_numpy(),
            storm_track["lon"].to_numpy(),
            storm_track["max_wind_kt"].to_numpy(),
            storm_track["timestamp"].to_numpy(),
        )

        above_threshold = max_wind >= min_wind_threshold_kt
        if not above_threshold.any():
            continue
        idx = np.where(above_threshold)[0]

        ratios = damage_ratio(max_wind[idx], construction_class[idx], roof_type[idx])
        significant = ratios >= min_damage_ratio_threshold
        if not significant.any():
            continue
        idx = idx[significant]
        ratios = ratios[significant]

        sub_df = pd.DataFrame(
            {
                "location_id": location_ids[idx],
                "storm_id": storm_id,
                "storm_name": storm_name,
                "loss_date": pd.to_datetime(loss_timestamps[idx]),
                "peril_type": "hurricane_wind",
                "max_wind_experienced_kt": max_wind[idx],
                "damage_ratio": ratios,
                "tiv_usd": tiv_usd[idx],
            }
        )
        sub_df = sub_df.merge(policy_lookup, on="location_id", how="inner")

        sub_df["incurred_loss_usd"] = (sub_df["damage_ratio"] * sub_df["tiv_usd"]).round(2)
        net_of_deductible = (sub_df["incurred_loss_usd"] - sub_df["deductible_usd"]).clip(lower=0)
        sub_df["paid_loss_usd"] = np.minimum(net_of_deductible, sub_df["limit_usd"]).round(2)

        result_frames.append(
            sub_df[
                [
                    "location_id",
                    "policy_id",
                    "storm_id",
                    "storm_name",
                    "loss_date",
                    "peril_type",
                    "max_wind_experienced_kt",
                    "damage_ratio",
                    "incurred_loss_usd",
                    "paid_loss_usd",
                ]
            ]
        )

        if (i + 1) % 50 == 0:
            running_total = sum(len(f) for f in result_frames)
            logger.info(
                "Processed %d/%d filtered storms, %d claim records so far",
                i + 1,
                len(storm_ids),
                running_total,
            )

    if not result_frames:
        return pd.DataFrame(
            columns=[
                "claim_id",
                "location_id",
                "policy_id",
                "storm_id",
                "storm_name",
                "loss_date",
                "peril_type",
                "max_wind_experienced_kt",
                "damage_ratio",
                "incurred_loss_usd",
                "paid_loss_usd",
            ]
        )

    result = pd.concat(result_frames, ignore_index=True)
    result.insert(0, "claim_id", [f"CLM{i:08d}" for i in range(1, len(result) + 1)])

    logger.info(
        "Generated %d total claim records across %d storms (of %d filtered near Florida)",
        len(result),
        result["storm_id"].nunique(),
        len(storm_ids),
    )
    return result