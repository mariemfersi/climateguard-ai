"""
Simplified parametric wind field model: estimates the wind speed
experienced at a given location during a hurricane, based on distance from
the storm's track and the storm's maximum sustained wind at each track
point.

METHODOLOGY NOTE — read before treating this as more sophisticated than it is:
Real production catastrophe models use physically-detailed parametric wind
field models (e.g., Holland 1980), which fit a pressure-profile "B
parameter" from central/ambient pressure and storm size, and explicitly
model wind field ASYMMETRY (stronger winds in the right-front quadrant
relative to storm motion). This module implements a much simpler
SYMMETRIC RADIAL POWER-LAW DECAY model:

    V(r) = Vmax                        for r <= Rmw
    V(r) = Vmax * (Rmw / r) ** decay_exponent   for r > Rmw

This is a standard simplified/pedagogical approximation of the real
physics (wind is strongest near the core and decays with distance), NOT a
literal reproduction of Holland's or any other specific published
peer-reviewed formula. In particular:
    - It ignores storm-motion asymmetry (real hurricanes are NOT radially
      symmetric — the right-front quadrant is meaningfully stronger).
    - `estimate_rmw_km()` uses an illustrative heuristic (smaller radius of
      maximum winds for more intense storms, which is physically correct
      in direction), not a fitted empirical relationship from a specific
      cited study.
    - `decay_exponent` is a fixed assumption (0.5), not calibrated per-storm.

This is an appropriate and clearly-documented simplification for a
portfolio-scale vertical slice. If used beyond that context, replace with
a real fitted parametric wind field model (e.g. via the `tcrm` or similar
open-source implementations of Holland's model).
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0
_DEFAULT_DECAY_EXPONENT = 0.5
_MIN_RMW_KM = 15.0
_MAX_RMW_KM = 60.0


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Great-circle distance in km between (lat1, lon1) and (lat2, lon2).
    Fully vectorized — accepts scalars or numpy arrays (with broadcasting).
    """
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_KM * c


def estimate_rmw_km(max_wind_kt: np.ndarray) -> np.ndarray:
    """
    Illustrative heuristic: more intense storms tend to have a tighter
    (smaller) radius of maximum winds. Direction is physically correct;
    exact values are a documented simplification, not a fitted formula.
    """
    max_wind_kt = np.asarray(max_wind_kt, dtype=float)
    rmw = 60.0 - 0.4 * (max_wind_kt - 50.0)
    return np.clip(rmw, _MIN_RMW_KM, _MAX_RMW_KM)


def wind_speed_at_distance(
    max_wind_kt: np.ndarray,
    distance_km: np.ndarray,
    rmw_km: np.ndarray,
    decay_exponent: float = _DEFAULT_DECAY_EXPONENT,
) -> np.ndarray:
    """
    Estimate wind speed (in the same units as max_wind_kt, i.e. knots) at
    `distance_km` from the storm center, given the storm's max wind and
    estimated radius of maximum winds.

    Vectorized: all array args must be broadcastable against each other.
    """
    max_wind_kt = np.asarray(max_wind_kt, dtype=float)
    distance_km = np.asarray(distance_km, dtype=float)
    rmw_km = np.asarray(rmw_km, dtype=float)

    # Avoid division-by-zero / negative-distance edge cases.
    safe_distance = np.maximum(distance_km, 1e-6)
    decayed = max_wind_kt * (rmw_km / safe_distance) ** decay_exponent

    return np.where(distance_km <= rmw_km, max_wind_kt, decayed)


def max_wind_experienced_by_locations(
    loc_lat: np.ndarray,
    loc_lon: np.ndarray,
    track_lat: np.ndarray,
    track_lon: np.ndarray,
    track_max_wind_kt: np.ndarray,
    max_influence_km: float = 400.0,
) -> np.ndarray:
    """
    For a set of N locations and a single storm's T track points, compute
    the peak wind speed each location experiences across the storm's full
    passage (i.e. the max over all track-point-induced wind estimates).

    This is the storm-passage simplification: rather than modeling exact
    timing/duration, we take the single worst instantaneous estimate per
    location, per storm — adequate for expected-annual-loss and VaR/TVaR
    estimation at the vertical-slice scale, though a production model would
    also track duration-of-exposure for demand-surge / duration-dependent
    damage effects.

    Args:
        loc_lat, loc_lon: shape (N,) location coordinates.
        track_lat, track_lon, track_max_wind_kt: shape (T,) storm track.
        max_influence_km: track points farther than this from a location
            are treated as contributing zero wind (avoids computing a tiny,
            physically-meaningless residual wind speed at extreme range).

    Returns:
        Array of shape (N,): peak experienced wind (knots) per location.
    """
    loc_lat = np.asarray(loc_lat, dtype=float).reshape(-1, 1)  # (N, 1)
    loc_lon = np.asarray(loc_lon, dtype=float).reshape(-1, 1)  # (N, 1)
    track_lat = np.asarray(track_lat, dtype=float).reshape(1, -1)  # (1, T)
    track_lon = np.asarray(track_lon, dtype=float).reshape(1, -1)  # (1, T)
    track_max_wind_kt = np.asarray(track_max_wind_kt, dtype=float).reshape(1, -1)  # (1, T)

    distance_km = haversine_km(loc_lat, loc_lon, track_lat, track_lon)  # (N, T)
    rmw_km = estimate_rmw_km(track_max_wind_kt)  # (1, T), broadcasts

    wind_at_each_point = wind_speed_at_distance(track_max_wind_kt, distance_km, rmw_km)  # (N, T)
    wind_at_each_point = np.where(distance_km <= max_influence_km, wind_at_each_point, 0.0)

    return wind_at_each_point.max(axis=1)  # (N,)