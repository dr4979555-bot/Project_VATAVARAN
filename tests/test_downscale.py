"""
Physics and GeoJSON validation tests for VATAWARAN Stage 2 Downscaling Engine.
Verifies mass divergence, moisture flux residuals, quantile monotonicity, and 5 km GeoJSON structure.
"""

from fastapi.testclient import TestClient
from backend.main import app, _generate_downscale_grid

client = TestClient(app)


def test_downscale_grid_generator_properties():
    """Verify _generate_downscale_grid produces 5 km point features with required properties."""
    lon, lat = 85.8, 20.2
    features = _generate_downscale_grid(
        center_lon=lon,
        center_lat=lat,
        hazard_type="TROPICAL_CYCLONE",
        grid_extent_km=200,
        resolution_km=5,
    )

    assert len(features) > 20
    for f in features:
        assert f["type"] == "Feature"
        assert f["geometry"]["type"] == "Point"
        coord = f["geometry"]["coordinates"]
        assert len(coord) == 2
        # Should be within ~2 degrees of center
        assert abs(coord[0] - lon) < 3.0
        assert abs(coord[1] - lat) < 3.0

        props = f["properties"]
        assert "precip_p50_mm" in props
        assert "precip_p90_mm" in props
        assert "precip_p99_mm" in props
        assert "wind_gust_kmh" in props
        assert "temperature_c" in props

        # Quantile monotonicity constraint
        assert props["precip_p50_mm"] <= props["precip_p90_mm"] <= props["precip_p99_mm"]
        assert props["wind_gust_kmh"] >= 0.0


def test_downscale_heatwave_physics():
    """Verify extreme heat generates dry conditions with elevated temperatures."""
    features = _generate_downscale_grid(
        center_lon=74.0,
        center_lat=28.0,
        hazard_type="EXTREME_HEAT",
    )
    temps = [f["properties"]["temperature_c"] for f in features]
    precips = [f["properties"]["precip_p50_mm"] for f in features]

    assert max(temps) >= 40.0  # Severe heatwave reaches > 40°C
    assert all(p == 0.0 for p in precips)  # Heatwave has no precipitation


def test_downscale_orographic_precipitation_physics():
    """Verify orographic rainfall features extreme localized precipitation."""
    features = _generate_downscale_grid(
        center_lon=79.0,
        center_lat=30.5,
        hazard_type="EXTREME_PRECIPITATION",
    )
    p99_vals = [f["properties"]["precip_p99_mm"] for f in features]
    assert max(p99_vals) > 100.0  # Extreme deluge exceeds 100 mm


def test_downscale_api_physics_residuals():
    """Verify downscale API response includes mass conservation and moisture flux QA checks."""
    payload = {
        "anomaly_id": "BOB-CYCLONE-2026-01",
        "lead_time_hours": 96,
        "target_variables": ["TP", "U10M", "V10M"],
        "num_ensemble_samples": 10,
    }
    response = client.post("/api/v1/downscale/generate", json=payload)
    assert response.status_code == 200
    data = response.json()

    metadata = data["metadata"]
    assert "physics_residuals" in metadata
    residuals = metadata["physics_residuals"]

    # Mass divergence residual bound (<= 0.05)
    assert "mass_divergence" in residuals
    assert 0.0 <= residuals["mass_divergence"] <= 0.05

    # Moisture flux residual bound (<= 0.15)
    assert "moisture_flux" in residuals
    assert 0.0 <= residuals["moisture_flux"] <= 0.15

    assert metadata["physics_qa_passed"] is True
    assert metadata["degraded_mode"] is False
    assert metadata["resolution_km"] == 5.0
