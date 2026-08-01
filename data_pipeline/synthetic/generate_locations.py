"""
Generate a population-weighted synthetic book of insured locations across
Florida / the Gulf Coast.

METHODOLOGY NOTE (read before treating this as authoritative):
The roadmap's original plan called for full US Census block-group
shapefiles as the population-density scaffold. This implementation uses a
lighter-weight, still-defensible substitute: a curated list of Florida's
major metro population centers (FLORIDA_METRO_CENTERS below) as weighted
sampling anchors, with locations scattered around each center via Gaussian
jitter scaled to that metro's approximate real geographic footprint.

This is a deliberate simplification, not a hidden one:
    - Population figures are approximate county/metro-area figures, rounded,
      used ONLY as relative sampling weights (i.e. "Miami-Dade gets roughly
      3x as many synthetic locations as Naples"), not presented as precise
      Census statistics anywhere downstream.
    - It avoids the heavy geospatial dependency chain (fiona, full TIGER/
      Line shapefile parsing, multi-hundred-MB downloads) that block-group-
      level precision would require, which is disproportionate for a
      hurricane-first vertical slice.
    - If block-group-level precision is later required (e.g. for the
      horizontal-expansion phase), swap this module's sampling logic for a
      real Census shapefile join — the downstream schema (location_id, lat,
      lon, ...) does not need to change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (name, lat, lon, approx_2020_census_population, jitter_std_degrees)
# Population figures are rounded approximations used only as sampling
# weights — see module docstring. jitter_std_degrees roughly reflects each
# metro area's real geographic spread (~1 degree latitude ~= 111 km).
FLORIDA_METRO_CENTERS = [
    ("Miami-Dade", 25.7617, -80.1918, 2_700_000, 0.12),
    ("Broward (Ft. Lauderdale)", 26.1224, -80.1373, 1_940_000, 0.10),
    ("Palm Beach (West Palm Beach)", 26.7153, -80.0534, 1_500_000, 0.12),
    ("Hillsborough (Tampa)", 27.9506, -82.4572, 1_460_000, 0.10),
    ("Pinellas (St. Petersburg)", 27.7676, -82.6403, 960_000, 0.08),
    ("Orange (Orlando)", 28.5383, -81.3792, 1_430_000, 0.10),
    ("Duval (Jacksonville)", 30.3322, -81.6557, 1_000_000, 0.10),
    ("Lee (Fort Myers)", 26.6406, -81.8723, 760_000, 0.09),
    ("Collier (Naples)", 26.1420, -81.7948, 375_000, 0.08),
    ("Sarasota", 27.3364, -82.5307, 434_000, 0.07),
    ("Cape Coral", 26.5629, -81.9495, 194_000, 0.06),
    ("St. Lucie (Port St. Lucie)", 27.2730, -80.3582, 205_000, 0.07),
    ("Volusia (Daytona Beach)", 29.2108, -81.0228, 553_000, 0.08),
    ("Escambia (Pensacola)", 30.4213, -87.2169, 322_000, 0.08),
    ("Bay (Panama City)", 30.1588, -85.6602, 175_000, 0.07),
    ("Monroe (Key West)", 24.5551, -81.7800, 83_000, 0.06),
]

# Rejects/redraws any sampled point outside this bounding box (avoids
# obviously-in-the-ocean points from an unlucky large jitter draw).
FLORIDA_BBOX = {"lat_min": 24.3, "lat_max": 31.1, "lon_min": -87.7, "lon_max": -79.7}

_MAX_REDRAW_ATTEMPTS = 20


def generate_locations(n: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate n population-weighted synthetic insured locations.

    Returns:
        DataFrame with columns [location_id, lat, lon, metro_center].

    Raises:
        ValueError: if n <= 0.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    rng = np.random.default_rng(seed)
    weights = np.array([c[3] for c in FLORIDA_METRO_CENTERS], dtype=float)
    probs = weights / weights.sum()

    center_idx = rng.choice(len(FLORIDA_METRO_CENTERS), size=n, p=probs)

    lats = np.empty(n)
    lons = np.empty(n)
    metro_names = np.empty(n, dtype=object)

    for i, idx in enumerate(center_idx):
        name, center_lat, center_lon, _, jitter_std = FLORIDA_METRO_CENTERS[idx]

        lat, lon = None, None
        for _attempt in range(_MAX_REDRAW_ATTEMPTS):
            candidate_lat = center_lat + rng.normal(0, jitter_std)
            candidate_lon = center_lon + rng.normal(0, jitter_std)
            if (
                FLORIDA_BBOX["lat_min"] <= candidate_lat <= FLORIDA_BBOX["lat_max"]
                and FLORIDA_BBOX["lon_min"] <= candidate_lon <= FLORIDA_BBOX["lon_max"]
            ):
                lat, lon = candidate_lat, candidate_lon
                break
        if lat is None:
            # Extremely unlikely given the bbox margins, but fail loudly
            # rather than silently placing a point outside Florida.
            raise RuntimeError(
                f"Failed to sample a valid location near '{name}' after "
                f"{_MAX_REDRAW_ATTEMPTS} attempts — check jitter_std / bbox."
            )

        lats[i] = lat
        lons[i] = lon
        metro_names[i] = name

    return pd.DataFrame(
        {
            "location_id": [f"LOC{i:07d}" for i in range(1, n + 1)],
            "lat": lats,
            "lon": lons,
            "metro_center": metro_names,
        }
    )