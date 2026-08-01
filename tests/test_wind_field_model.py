"""
Unit tests for data_pipeline.synthetic.wind_field_model.

These deliberately check PHYSICAL SENSIBILITY (wind decays with distance,
core wind is roughly constant within Rmw, a location directly under the
storm sees max_wind_kt exactly) rather than just schema/shape correctness
— consistent with the physical-sensibility checks used for HURDAT2 parsing.
"""

import numpy as np
import pytest

from data_pipeline.synthetic.wind_field_model import (
    estimate_rmw_km,
    haversine_km,
    max_wind_experienced_by_locations,
    wind_speed_at_distance,
)


# --- haversine_km ---------------------------------------------------------


def test_haversine_zero_distance_for_identical_points():
    assert haversine_km(25.0, -80.0, 25.0, -80.0) == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance_miami_to_orlando():
    """Miami (25.7617, -80.1918) to Orlando (28.5383, -81.3792) is a
    real, well-known distance of approximately 330-400 km."""
    dist = haversine_km(25.7617, -80.1918, 28.5383, -81.3792)
    assert 330 <= dist <= 400


def test_haversine_is_symmetric():
    d1 = haversine_km(25.0, -80.0, 28.0, -82.0)
    d2 = haversine_km(28.0, -82.0, 25.0, -80.0)
    assert d1 == pytest.approx(d2)


def test_haversine_vectorized_broadcasting():
    lats1 = np.array([[25.0], [26.0], [27.0]])  # (3, 1)
    lons1 = np.array([[-80.0], [-81.0], [-82.0]])
    lats2 = np.array([[25.0, 26.0]])  # (1, 2)
    lons2 = np.array([[-80.0, -81.0]])

    result = haversine_km(lats1, lons1, lats2, lons2)
    assert result.shape == (3, 2)
    assert result[0, 0] == pytest.approx(0.0, abs=1e-6)  # same point


# --- estimate_rmw_km --------------------------------------------------------


def test_estimate_rmw_more_intense_storms_have_smaller_radius():
    """Core physical claim of this heuristic: stronger storms -> tighter
    wind field. This must hold or the whole model direction is wrong."""
    rmw_cat1 = estimate_rmw_km(75)  # ~Cat 1
    rmw_cat5 = estimate_rmw_km(160)  # ~Cat 5
    assert rmw_cat5 < rmw_cat1


def test_estimate_rmw_bounded():
    rmw = estimate_rmw_km(np.array([30, 80, 130, 200]))
    assert (rmw >= 15.0).all()
    assert (rmw <= 60.0).all()


# --- wind_speed_at_distance --------------------------------------------------


def test_wind_speed_at_zero_distance_equals_max_wind():
    result = wind_speed_at_distance(max_wind_kt=100, distance_km=0, rmw_km=30)
    assert result == pytest.approx(100.0)


def test_wind_speed_constant_within_rmw():
    """Within the radius of maximum winds, the simplified model treats
    wind speed as roughly constant (eyewall region)."""
    result_near_core = wind_speed_at_distance(max_wind_kt=100, distance_km=10, rmw_km=30)
    result_at_rmw = wind_speed_at_distance(max_wind_kt=100, distance_km=30, rmw_km=30)
    assert result_near_core == pytest.approx(100.0)
    assert result_at_rmw == pytest.approx(100.0)


def test_wind_speed_decays_beyond_rmw():
    result_at_rmw = wind_speed_at_distance(max_wind_kt=100, distance_km=30, rmw_km=30)
    result_far = wind_speed_at_distance(max_wind_kt=100, distance_km=200, rmw_km=30)
    assert result_far < result_at_rmw
    assert result_far > 0  # decays but never goes negative


def test_wind_speed_monotonically_decreasing_with_distance():
    distances = np.array([30, 60, 100, 200, 400])
    speeds = wind_speed_at_distance(max_wind_kt=100, distance_km=distances, rmw_km=30)
    assert (np.diff(speeds) <= 0).all()  # monotonically non-increasing


# --- max_wind_experienced_by_locations --------------------------------------


def test_location_directly_on_track_sees_full_max_wind():
    """A location exactly at a track point's coordinates should experience
    (approximately) that track point's full max wind."""
    loc_lat, loc_lon = np.array([25.0]), np.array([-80.0])
    track_lat = np.array([24.0, 25.0, 26.0])
    track_lon = np.array([-81.0, -80.0, -79.0])
    track_wind = np.array([80.0, 120.0, 90.0])

    result = max_wind_experienced_by_locations(loc_lat, loc_lon, track_lat, track_lon, track_wind)
    assert result[0] == pytest.approx(120.0, rel=0.01)


def test_distant_location_experiences_negligible_wind():
    """A location far outside max_influence_km from every track point
    should experience ~zero wind, not a spurious residual value."""
    loc_lat, loc_lon = np.array([40.0]), np.array([-70.0])  # far from Florida track
    track_lat = np.array([25.0, 26.0])
    track_lon = np.array([-80.0, -81.0])
    track_wind = np.array([100.0, 100.0])

    result = max_wind_experienced_by_locations(
        loc_lat, loc_lon, track_lat, track_lon, track_wind, max_influence_km=400.0
    )
    assert result[0] == pytest.approx(0.0)


def test_closer_location_experiences_more_wind_than_farther_location():
    track_lat = np.array([25.0])
    track_lon = np.array([-80.0])
    track_wind = np.array([100.0])

    close_loc = max_wind_experienced_by_locations(
        np.array([25.1]), np.array([-80.1]), track_lat, track_lon, track_wind
    )
    far_loc = max_wind_experienced_by_locations(
        np.array([27.0]), np.array([-82.0]), track_lat, track_lon, track_wind
    )
    assert close_loc[0] > far_loc[0]


def test_output_shape_matches_number_of_locations():
    n_locations = 50
    loc_lat = np.random.uniform(25, 30, n_locations)
    loc_lon = np.random.uniform(-85, -80, n_locations)
    track_lat = np.array([25.0, 26.0, 27.0])
    track_lon = np.array([-80.0, -81.0, -82.0])
    track_wind = np.array([90.0, 100.0, 95.0])

    result = max_wind_experienced_by_locations(loc_lat, loc_lon, track_lat, track_lon, track_wind)
    assert result.shape == (n_locations,)