"""
Unit and integration tests for VATAWARAN FastAPI endpoints.
Verifies TRD §5 contract specifications and government integration endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_system_status():
    """Verify /api/v1/system/status returns operational status and all 5 pipeline stages."""
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert data["uptime_seconds"] > 0
    assert "gpu_available" in data
    assert "forecast_cycle" in data
    assert len(data["pipeline_stages"]) >= 5
    assert data["gov_sources_active"] >= 4

    stage_names = [s["name"] for s in data["pipeline_stages"]]
    assert any("Ingestion" in name for name in stage_names)
    assert any("GNN" in name for name in stage_names)
    assert any("Diffusion" in name for name in stage_names)


def test_forecast_track_default():
    """Verify /api/v1/forecast/track returns active extreme weather anomalies."""
    response = client.get("/api/v1/forecast/track")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["anomalies_detected"] > 0
    assert len(data["data"]) > 0

    first = data["data"][0]
    assert "anomaly_id" in first
    assert "hazard_type" in first
    assert "peak_efi" in first
    assert len(first["trajectories"]) > 0

    traj = first["trajectories"][0]
    assert "lead_time_hours" in traj
    assert "valid_utc" in traj
    assert len(traj["centroid"]) == 2
    assert len(traj["bounding_box"]) == 4
    assert 0.0 <= traj["efi_score"] <= 1.0


def test_forecast_track_filter():
    """Verify lead time and EFI filtering on /api/v1/forecast/track."""
    response = client.get("/api/v1/forecast/track?lead_time_min=72&lead_time_max=120&min_efi=0.88")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    for anomaly in data["data"]:
        assert anomaly["peak_efi"] >= 0.88
        for t in anomaly["trajectories"]:
            assert 72 <= t["lead_time_hours"] <= 120
            assert t["efi_score"] >= 0.88


def test_downscale_generate_success():
    """Verify /api/v1/downscale/generate generates 5 km resolution GeoJSON with physics QA."""
    payload = {
        "anomaly_id": "BOB-CYCLONE-2026-01",
        "lead_time_hours": 96,
        "target_variables": ["TP", "U10M", "V10M"],
        "num_ensemble_samples": 10,
    }
    response = client.post("/api/v1/downscale/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert "metadata" in data
    assert data["metadata"]["anomaly_id"] == "BOB-CYCLONE-2026-01"
    assert data["metadata"]["resolution_km"] == 5.0
    assert data["metadata"]["physics_qa_passed"] is True
    assert "mass_divergence" in data["metadata"]["physics_residuals"]
    assert "moisture_flux" in data["metadata"]["physics_residuals"]

    # Verify features exist (alert centroid + 5km grid points)
    features = data["features"]
    assert len(features) > 10

    # First feature is the impact alert
    alert = features[0]
    assert alert["properties"]["alert_type"] == "SEVERE_WEATHER_IMPACT"
    assert alert["properties"]["impact_radius_km"] == 5.0
    assert "metrics" in alert["properties"]

    # Subsequent features are grid points
    grid_point = features[1]
    assert grid_point["geometry"]["type"] == "Point"
    assert len(grid_point["geometry"]["coordinates"]) == 2
    props = grid_point["properties"]
    assert "precip_p50_mm" in props
    assert "precip_p90_mm" in props
    assert "precip_p99_mm" in props
    assert "wind_gust_kmh" in props
    assert "temperature_c" in props


def test_downscale_generate_unknown_anomaly():
    """Verify downscale endpoint returns error status for unknown anomaly ID."""
    response = client.post("/api/v1/downscale/generate", json={"anomaly_id": "UNKNOWN-999"})
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "error"
    assert "Unknown anomaly_id" in data.get("message", "")


def test_gov_sources_status():
    """Verify /api/v1/gov/sources returns metadata for all 4 Indian meteorological sources."""
    response = client.get("/api/v1/gov/sources")
    assert response.status_code == 200
    sources = response.json()
    assert len(sources) >= 4

    source_ids = {s["source_id"] for s in sources}
    assert "MOES-NCMRWF-NEPSG" in source_ids
    assert "MOES-IMD-AWS" in source_ids
    assert "ISRO-SAC-MOSDAC" in source_ids
    assert "MHA-NDMA-SACHET" in source_ids

    for s in sources:
        assert s["status"] in ["ONLINE", "SYNCED", "DEGRADED", "STANDBY"]
        assert s["latency_ms"] >= 0
        assert len(s["variables"]) > 0


def test_gov_ncmrwf_cycle():
    """Verify /api/v1/gov/ncmrwf/cycle returns NEPS-G 12 km cycle metadata."""
    response = client.get("/api/v1/gov/ncmrwf/cycle")
    assert response.status_code == 200
    data = response.json()
    assert "NEPS-G" in data["model"]
    assert data["grid_resolution_km"] == 12.0
    assert data["ensemble_members"] == 50
    assert len(data["forecast_lead_steps_hours"]) > 0


def test_gov_imd_stations():
    """Verify /api/v1/gov/imd/stations returns calibrated ground telemetry."""
    response = client.get("/api/v1/gov/imd/stations")
    assert response.status_code == 200
    stations = response.json()
    assert len(stations) >= 10

    for st in stations:
        assert "station_id" in st
        assert "station_name" in st
        assert "temp_c" in st
        assert 0.0 <= st["humidity_pct"] <= 100.0
        assert st["rain_24h_mm"] >= 0.0
        assert st["wind_speed_kmh"] >= 0.0
        assert 0 <= st["wind_direction_deg"] <= 360
        assert st["status"] == "CALIBRATED_ONLINE"


def test_gov_mosdac_satellite():
    """Verify /api/v1/gov/mosdac/satellite returns INSAT-3DR and 3DS products."""
    response = client.get("/api/v1/gov/mosdac/satellite")
    assert response.status_code == 200
    products = response.json()
    assert len(products) >= 2

    satellites = {p["satellite"] for p in products}
    assert "INSAT-3DR" in satellites
    assert "INSAT-3DS" in satellites


def test_gov_cap_alerts():
    """Verify /api/v1/gov/cap/alerts generates NDMA CAP alert payloads for all anomalies."""
    response = client.get("/api/v1/gov/cap/alerts?lead_time_hours=96")
    assert response.status_code == 200
    alerts = response.json()
    assert len(alerts) >= 3

    for alert in alerts:
        assert alert["identifier"].startswith("IN-NDMA-VATAWARAN-")
        assert alert["status"] == "Actual"
        assert alert["msg_type"] == "Alert"
        assert alert["severity"] in ["EXTREME", "SEVERE", "HIGH"]
        assert len(alert["circle"].split(",")) == 3


def test_gov_cap_xml_export_success():
    """Verify /api/v1/gov/cap/export/{id} exports valid OASIS CAP v1.2 XML."""
    response = client.get("/api/v1/gov/cap/export/BOB-CYCLONE-2026-01?lead_time_hours=96")
    assert response.status_code == 200
    assert "application/xml" in response.headers.get("content-type", "")
    xml_text = response.text
    assert "<alert" in xml_text
    assert "urn:oasis:names:tc:emergency:cap:1.2" in xml_text
    assert "BOB-CYCLONE-2026-01" in xml_text
    assert "<instruction>" in xml_text
    assert "<circle>" in xml_text


def test_gov_cap_xml_export_404():
    """Verify /api/v1/gov/cap/export/{id} returns 404 for nonexistent anomaly."""
    response = client.get("/api/v1/gov/cap/export/DOES-NOT-EXIST")
    assert response.status_code == 404


def test_frontend_serving():
    """Verify root / serves HTML dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
