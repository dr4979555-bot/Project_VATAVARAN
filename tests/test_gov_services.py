"""
Unit tests for VATAWARAN Indian Government meteorological & disaster management services.
Verifies NCMRWF, IMD, MOSDAC, and NDMA ITU-T X.1303 CAP v1.2 XML compliance.
"""

from xml.etree import ElementTree as ET

from backend.gov_services import (
    NCMRWFService,
    IMDService,
    MOSDACService,
    NDMACAPService,
    gov_hub,
)


def test_ncmrwf_cycle_metadata():
    """Verify NCMRWF service returns expected NEPS-G 12 km grid and baseline."""
    service = NCMRWFService()
    cycle = service.get_latest_cycle()

    assert "NEPS-G" in cycle["model"]
    assert cycle["grid_resolution_km"] == 12.0
    assert cycle["ensemble_members"] == 50
    assert 850 in cycle["pressure_levels"]
    assert 500 in cycle["pressure_levels"]
    assert cycle["status"] == "ONLINE"
    assert "IMDAA" in cycle["climatological_baseline"]


def test_imd_station_registry_and_telemetry():
    """Verify IMD ground station network contains georeferenced stations with valid sensors."""
    service = IMDService()
    assert len(service.STATION_REGISTRY) >= 14

    for st in service.STATION_REGISTRY:
        assert st["id"].startswith("IMD_")
        assert len(st["name"]) > 0
        assert 8.0 <= st["lat"] <= 38.0  # Indian latitude bounds
        assert 68.0 <= st["lon"] <= 98.0  # Indian longitude bounds
        assert st["elev"] >= 0

    obs = service.get_realtime_observations()
    assert len(obs) == len(service.STATION_REGISTRY)

    for o in obs:
        assert -10.0 <= o.temp_c <= 60.0
        assert 0.0 <= o.humidity_pct <= 100.0
        assert o.rain_24h_mm >= 0.0
        assert o.wind_speed_kmh >= 0.0
        assert 0 <= o.wind_direction_deg <= 360
        assert 800.0 <= o.pressure_hpa <= 1050.0
        assert o.status == "CALIBRATED_ONLINE"


def test_mosdac_satellite_products():
    """Verify MOSDAC service provides INSAT-3DR and 3DS products with valid geophysical limits."""
    service = MOSDACService()
    products = service.get_latest_satellite_feed()

    assert len(products) >= 2
    for p in products:
        assert p.satellite in ["INSAT-3DR", "INSAT-3DS"]
        assert p.resolution_km > 0.0
        assert p.cloud_top_temp_c <= 0.0  # Convective cloud tops are sub-zero
        assert p.olr_wm2 > 50.0  # Standard Outgoing Longwave Radiation range
        assert p.data_url.startswith("https://")


def test_ndma_cap_alert_builder():
    """Verify CAP alert generation across multiple hazard types."""
    hazards = ["TROPICAL_CYCLONE", "EXTREME_HEAT", "EXTREME_PRECIPITATION"]

    for hazard in hazards:
        alert = NDMACAPService.build_cap_alert(
            anomaly_id=f"TEST-{hazard}",
            hazard_type=hazard,
            lead_time_hours=48,
            centroid=[85.8, 20.2],
            severity_level="EXTREME",
            metrics={"peak_efi": 0.95, "confidence_score": 0.92},
            advisory="Test extreme hazard warning advisory.",
            valid_utc="2026-10-02T12:00:00Z",
        )

        assert alert.identifier.startswith(f"IN-NDMA-VATAWARAN-TEST-{hazard}")
        assert alert.sender == "vatawaran.engine@ncmrwf.gov.in"
        assert alert.category == "Met"
        assert alert.severity == "EXTREME"
        assert alert.scope == "Public"
        assert alert.status == "Actual"
        assert alert.msg_type == "Alert"
        assert len(alert.instruction) > 20
        assert len(alert.headline) > 10

        # Circle must be lat,lon,radius_km
        parts = alert.circle.split(",")
        assert len(parts) == 3
        lat, lon, radius = float(parts[0]), float(parts[1]), float(parts[2])
        assert round(lat, 1) == 20.2
        assert round(lon, 1) == 85.8
        assert radius == 5.0


def test_ndma_cap_xml_conformance():
    """Verify generated CAP XML strictly adheres to OASIS / ITU-T X.1303 specification."""
    alert = NDMACAPService.build_cap_alert(
        anomaly_id="BOB-CYCLONE-2026-01",
        hazard_type="TROPICAL_CYCLONE",
        lead_time_hours=96,
        centroid=[87.80, 19.80],
        severity_level="EXTREME",
        metrics={"peak_efi": 0.94, "confidence_score": 0.94},
        advisory="Intense tropical cyclone.",
        valid_utc="2026-10-04T00:00:00Z",
    )

    xml_str = NDMACAPService.generate_cap_xml(alert)

    # Must parse without syntax errors
    root = ET.fromstring(xml_str)
    ns = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}

    assert root.tag == "{urn:oasis:names:tc:emergency:cap:1.2}alert"

    # Core elements
    identifier = root.find("cap:identifier", ns)
    sender = root.find("cap:sender", ns)
    status = root.find("cap:status", ns)
    info = root.find("cap:info", ns)

    assert identifier is not None and identifier.text == alert.identifier
    assert sender is not None and "ncmrwf" in sender.text
    assert status is not None and status.text == "Actual"
    assert info is not None

    # Info subelements
    event = info.find("cap:event", ns)
    urgency = info.find("cap:urgency", ns)
    severity = info.find("cap:severity", ns)
    certainty = info.find("cap:certainty", ns)
    area = info.find("cap:area", ns)

    assert event is not None
    assert urgency is not None
    assert severity is not None and severity.text == "EXTREME"
    assert certainty is not None
    assert area is not None

    # Area subelements
    circle = area.find("cap:circle", ns)
    assert circle is not None and circle.text == "19.8000,87.8000,5.0"


def test_gov_hub_aggregation():
    """Verify gov_hub aggregates all sources with expected status."""
    statuses = gov_hub.get_all_source_statuses()
    assert len(statuses) >= 4
    for st in statuses:
        assert st.latency_ms >= 0
        assert len(st.endpoint) > 0
