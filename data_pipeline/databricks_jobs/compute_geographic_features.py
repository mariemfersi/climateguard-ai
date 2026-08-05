"""
Compute static geographic risk features for locations.

This module adds legitimate, knowable-at-pricing-time geographic features
like distance to coast, which are standard in catastrophe modeling.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Approximate Gulf Coast coastline points (simplified polygon)
# This is a rough approximation - in production, use a proper coastline shapefile
GULF_COAST_COASTLINE = [
    (30.0, -88.0),   # Mobile Bay area
    (30.3, -87.5),   # Alabama coast
    (30.4, -87.0),   # Florida Panhandle
    (30.2, -86.5),   # Panama City
    (29.9, -85.5),   # Apalachicola
    (29.8, -84.5),   # Big Bend
    (29.6, -83.5),   # Cedar Key
    (29.2, -82.5),   # Tampa Bay
    (28.5, -82.0),   # Sarasota
    (27.8, -82.0),   # Fort Myers
    (26.5, -82.0),   # Naples
    (25.5, -81.0),   # Everglades
    (25.2, -80.5),   # Miami
    (25.0, -80.0),   # Florida Keys
    (24.5, -81.5),   # Lower Keys
    (25.0, -97.0),   # Texas (Corpus Christi)
    (26.0, -97.0),   # Texas
    (27.0, -97.5),   # Texas
    (28.0, -96.5),   # Texas
    (29.0, -95.0),   # Texas (Galveston)
    (29.8, -94.0),   # Texas
    (30.0, -93.5),   # Louisiana
    (29.5, -92.0),   # Louisiana
    (29.0, -90.0),   # Mississippi
]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two points using the Haversine formula.
    
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


def compute_distance_to_coast(locations: pd.DataFrame) -> pd.DataFrame:
    """
    Compute minimum distance from each location to the Gulf Coast coastline.
    
    Args:
        locations: DataFrame with location_id, lat, lon columns
    
    Returns:
        DataFrame with location_id and distance_to_coast_km columns
    """
    coastline_array = np.array(GULF_COAST_COASTLINE)
    coastline_lats = coastline_array[:, 0]
    coastline_lons = coastline_array[:, 1]
    
    distances = []
    
    for _, row in locations.iterrows():
        loc_lat = row["lat"]
        loc_lon = row["lon"]
        
        # Compute distance to all coastline points
        point_distances = [
            haversine_distance(loc_lat, loc_lon, coast_lat, coast_lon)
            for coast_lat, coast_lon in GULF_COAST_COASTLINE
        ]
        
        min_distance = min(point_distances)
        distances.append(min_distance)
    
    result = pd.DataFrame({
        "location_id": locations["location_id"],
        "distance_to_coast_km": distances,
    })
    
    return result


def compute_geographic_risk_features(locations: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all static geographic risk features for locations.
    
    Args:
        locations: DataFrame with location_id, lat, lon columns
    
    Returns:
        DataFrame with location_id and geographic risk features
    """
    logger.info("Computing distance to coast...")
    coast_distances = compute_distance_to_coast(locations)
    
    logger.info(f"Distance to coast stats: {coast_distances['distance_to_coast_km'].describe()}")
    
    return coast_distances
