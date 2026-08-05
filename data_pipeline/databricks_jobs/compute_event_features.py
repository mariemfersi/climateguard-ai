"""
Compute event-focused climate features using HURDAT2 storm track data.

This module replaces simple yearly climate averages with event-focused features
that capture the actual hurricane risk experienced at each location-year:
- Storm count affecting each location
- Minimum distance to hurricane tracks
- Maximum sustained wind from nearby storms
- Number of days above wind thresholds
- Minimum pressure during storm events
- Extreme value statistics

These features are more predictive of actual hurricane risk than simple
climatological averages because they capture the specific storm activity
that affected each location in each year.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# Wind speed thresholds (knots) for counting extreme events
WIND_THRESHOLDS_KT = [34, 50, 64, 100]  # TS, Storm, Cat 1, Cat 3+
# Distance threshold (km) for considering a storm as "affecting" a location
STORM_INFLUENCE_RADIUS_KM = 500


def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Compute great-circle distance between points using the Haversine formula.
    
    Args:
        lat1, lon1: First point coordinates in degrees
        lat2, lon2: Second point coordinates in degrees
    
    Returns:
        Distance in kilometers
    """
    R = 6371.0  # Earth radius in km
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def compute_storm_location_distances(
    storm_tracks: pd.DataFrame,
    locations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute minimum distance from each location to each storm's track.
    
    Uses cKDTree for efficient spatial queries.
    
    Args:
        storm_tracks: HURDAT2 storm observations with lat, lon, timestamp, storm_id
        locations: Location data with location_id, lat, lon
    
    Returns:
        DataFrame with columns: [location_id, storm_id, min_distance_km, 
                               max_wind_at_closest_kt, min_pressure_at_closest_mb]
    """
    # Build spatial index for storm track points
    storm_coords = storm_tracks[["lat", "lon"]].values
    tree = cKDTree(storm_coords)
    
    location_coords = locations[["lat", "lon"]].values
    
    # Query for nearest storm point for each location
    distances, indices = tree.query(location_coords, k=1)
    
    # Get the storm information for the closest points
    closest_points = storm_tracks.iloc[indices].reset_index(drop=True)
    
    result = pd.DataFrame({
        "location_id": locations["location_id"].values[min(locations.index, len(closest_points))],
        "storm_id": closest_points["storm_id"].values,
        "min_distance_km": distances,
        "max_wind_at_closest_kt": closest_points["max_wind_kt"].values,
        "min_pressure_at_closest_mb": closest_points["min_pressure_mb"].values,
    })
    
    return result


def compute_yearly_event_features(
    storm_tracks: pd.DataFrame,
    locations: pd.DataFrame,
    years: list[int],
    influence_radius_km: float = STORM_INFLUENCE_RADIUS_KM,
) -> pd.DataFrame:
    """
    Compute event-focused features for each (location, year) pair.
    
    Args:
        storm_tracks: HURDAT2 storm observations with lat, lon, timestamp, 
                     storm_id, max_wind_kt, min_pressure_mb
        locations: Location data with location_id, lat, lon
        years: List of years to compute features for
        influence_radius_km: Distance threshold for considering a storm as affecting
    
    Returns:
        DataFrame with one row per (location_id, year) and event-focused features
    """
    # Add year to storm tracks
    storm_tracks = storm_tracks.copy()
    storm_tracks["year"] = storm_tracks["timestamp"].dt.year
    
    # Filter to relevant years
    storm_tracks = storm_tracks[storm_tracks["year"].isin(years)]
    
    rows = []
    
    for year in years:
        year_storms = storm_tracks[storm_tracks["year"] == year]
        
        if year_storms.empty:
            # No storms this year - all locations get zero features
            for _, loc in locations.iterrows():
                rows.append({
                    "location_id": loc["location_id"],
                    "year": year,
                    "storm_count": 0,
                    "min_distance_to_track_km": np.nan,
                    "max_wind_nearby_kt": 0,
                    "min_pressure_nearby_mb": np.nan,
                    "days_above_34kt": 0,
                    "days_above_50kt": 0,
                    "days_above_64kt": 0,
                    "days_above_100kt": 0,
                })
            continue
        
        # For each location, compute storm proximity features
        for _, loc in locations.iterrows():
            loc_lat, loc_lon = loc["lat"], loc["lon"]
            
            # Compute distances to all storm track points this year
            distances = haversine_distance(
                np.full(len(year_storms), loc_lat),
                np.full(len(year_storms), loc_lon),
                year_storms["lat"].values,
                year_storms["lon"].values,
            )
            
            # Find storms within influence radius
            nearby_mask = distances <= influence_radius_km
            nearby_storms = year_storms[nearby_mask]
            
            if nearby_storms.empty:
                # No storms nearby
                rows.append({
                    "location_id": loc["location_id"],
                    "year": year,
                    "storm_count": 0,
                    "min_distance_to_track_km": np.nan,
                    "max_wind_nearby_kt": 0,
                    "min_pressure_nearby_mb": np.nan,
                    "days_above_34kt": 0,
                    "days_above_50kt": 0,
                    "days_above_64kt": 0,
                    "days_above_100kt": 0,
                })
            else:
                # Compute features for nearby storms
                unique_storms = nearby_storms["storm_id"].nunique()
                min_dist = distances[nearby_mask].min()
                max_wind = nearby_storms["max_wind_kt"].max()
                min_pressure = nearby_storms["min_pressure_mb"].min()
                
                # Count days above wind thresholds
                days_above = {}
                for threshold in WIND_THRESHOLDS_KT:
                    days_above[f"days_above_{threshold}kt"] = (
                        (nearby_storms["max_wind_kt"] >= threshold).sum()
                    )
                
                rows.append({
                    "location_id": loc["location_id"],
                    "year": year,
                    "storm_count": unique_storms,
                    "min_distance_to_track_km": min_dist,
                    "max_wind_nearby_kt": max_wind,
                    "min_pressure_nearby_mb": min_pressure,
                    **days_above,
                })
    
    return pd.DataFrame(rows)


def load_hurdat2_tracks(bronze_path: Path = Path("data_pipeline/bronze/hurdat2/tracks.parquet")) -> pd.DataFrame:
    """
    Load HURDAT2 storm tracks from Bronze layer.
    
    Args:
        bronze_path: Path to HURDAT2 parquet file
    
    Returns:
        DataFrame with storm track observations
    """
    if not bronze_path.exists():
        raise FileNotFoundError(f"HURDAT2 data not found at {bronze_path}")
    
    return pd.read_parquet(bronze_path)


def compute_regional_encoding(
    locations: pd.DataFrame,
    n_clusters: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute regional encodings for lat/lon to reduce geographic memorization.
    
    Uses K-means clustering to group locations into regions.
    
    Args:
        locations: Location data with location_id, lat, lon
        n_clusters: Number of regional clusters
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with location_id and region_cluster columns
    """
    from sklearn.cluster import KMeans
    
    coords = locations[["lat", "lon"]].values
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    regions = kmeans.fit_predict(coords)
    
    result = pd.DataFrame({
        "location_id": locations["location_id"],
        "region_cluster": regions,
    })
    
    return result


def validate_data_alignment(
    locations: pd.DataFrame,
    claims: pd.DataFrame,
    event_features: pd.DataFrame,
) -> dict:
    """
    Validate that climate data, claims, and exposure are correctly aligned
    by both location and year.
    
    Args:
        locations: Location data with location_id
        claims: Claims data with location_id, loss_date
        event_features: Event features with location_id, year
    
    Returns:
        Dictionary with validation results
    """
    results = {}
    
    # Check location coverage
    loc_ids = set(locations["location_id"])
    claim_loc_ids = set(claims["location_id"])
    event_loc_ids = set(event_features["location_id"])
    
    results["locations_in_exposure"] = len(loc_ids)
    results["locations_in_claims"] = len(claim_loc_ids)
    results["locations_in_event_features"] = len(event_loc_ids)
    
    results["missing_from_claims"] = len(loc_ids - claim_loc_ids)
    results["missing_from_event_features"] = len(loc_ids - event_loc_ids)
    
    # Check year alignment
    claims["year"] = pd.to_datetime(claims["loss_date"]).dt.year
    claim_years = set(claims["year"].unique())
    event_years = set(event_features["year"].unique())
    
    results["years_in_claims"] = len(claim_years)
    results["years_in_event_features"] = len(event_years)
    results["year_overlap"] = len(claim_years & event_years)
    
    # Check for orphaned records
    orphaned_claims = claims[~claims["location_id"].isin(loc_ids)]
    results["orphaned_claims"] = len(orphaned_claims)
    
    orphaned_events = event_features[~event_features["location_id"].isin(loc_ids)]
    results["orphaned_event_features"] = len(orphaned_events)
    
    return results
