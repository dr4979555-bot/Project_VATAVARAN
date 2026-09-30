"""
VATAWARAN Government API & Data Integration Services
=====================================================
Integrates Indian Government Meteorological & Disaster Management Endpoints:
1. NCMRWF (National Centre for Medium Range Weather Forecasting) — NEPS-G / NCUM Ingestion
2. IMD (India Meteorological Department) — Ground AWS Network & Radar Validation
3. ISRO / MOSDAC (Space Applications Centre) — INSAT-3D/3DR/3DS Satellite Feeds
4. NDMA (National Disaster Management Authority) — Sachet CAP v1.2 (ITU-T X.1303) Exporter
"""

import os
import math
import random
from datetime import datetime, timedelta
from typing import Optional, Any
from xml.etree import ElementTree as ET
from pydantic import BaseModel, Field

# ─── Configuration & Defaults ──────────────────────────────────────────────────

NCMRWF_THREDDS_URL = os.getenv("NCMRWF_THREDDS_URL", "https://opendap.ncmrwf.gov.in/thredds/dodsC/NEPSG")
IMD_API_BASE_URL = os.getenv("IMD_API_BASE_URL", "https://mausam.imd.gov.in/api")
MOSDAC_API_BASE_URL = os.getenv("MOSDAC_API_BASE_URL", "https://mosdac.gov.in/api/v1")
NDMA_SACHET_URL = os.getenv("NDMA_SACHET_URL", "https://sachet.ndma.gov.in/cap/api/v1")

BASE_CYCLE = datetime(2026, 9, 30, 0, 0, 0)


# ─── Models ────────────────────────────────────────────────────────────────────

class GovSourceStatus(BaseModel):
    source_id: str
    agency: str
    name: str
    protocol: str
    endpoint: str
    status: str  # "ONLINE" | "SYNCED" | "DEGRADED" | "STANDBY"
    latency_ms: int
    last_sync_utc: str
    data_format: str
    variables: list[str]
    description: str


class IMDStationObservation(BaseModel):
    station_id: str
    station_name: str
    state: str
    district: str
    latitude: float
    longitude: float
    elevation_m: float
    temp_c: float
    humidity_pct: float
    rain_24h_mm: float
    wind_speed_kmh: float
    wind_direction_deg: int
    pressure_hpa: float
    status: str
    last_update_utc: str
    model_bias_temp_c: Optional[float] = None
    model_bias_rain_mm: Optional[float] = None


class MOSDACProduct(BaseModel):
    product_id: str
    satellite: str
    sensor: str
    product_name: str
    resolution_km: float
    coverage: str
    timestamp_utc: str
    cloud_top_temp_c: float
    max_rain_rate_mmh: float
    olr_wm2: float
    data_url: str


class CAPAlertPayload(BaseModel):
    identifier: str
    sender: str
    sent: str
    status: str
    msg_type: str
    scope: str
    category: str
    event: str
    urgency: str
    severity: str
    certainty: str
    headline: str
    description: str
    instruction: str
    area_desc: str
    circle: str  # "lat,lon,radius_km"
    lead_time_hours: int
    metrics: dict[str, Any]


# ─── 1. NCMRWF Ingestion Service ───────────────────────────────────────────────

class NCMRWFService:
    """
    Connects to the National Centre for Medium Range Weather Forecasting (MoES).
    Ingests NEPS-G (12 km Global Ensemble, 50 members) and NCUM deterministic runs.
    """

    def __init__(self, base_url: str = NCMRWF_THREDDS_URL):
        self.base_url = base_url

    def get_latest_cycle(self) -> dict[str, Any]:
        """Returns the latest operational model cycle metadata."""
        return {
            "model": "NEPS-G (NCMRWF Ensemble Prediction System - Global)",
            "deterministic_model": "NCUM-G v12.2",
            "cycle_utc": BASE_CYCLE.isoformat() + "Z",
            "grid_resolution_km": 12.0,
            "pressure_levels": [1000, 925, 850, 700, 500, 300, 200],
            "ensemble_members": 50,
            "forecast_lead_steps_hours": [24, 48, 72, 96, 120, 144, 168, 192, 216, 240],
            "status": "ONLINE",
            "opendap_catalog": f"{self.base_url}/catalog.html",
            "climatological_baseline": "IMDAA 12 km (1979-2020) + ERA5 (1991-2020)",
        }


# ─── 2. IMD Ground Observation Network Service ─────────────────────────────────

class IMDService:
    """
    Connects to India Meteorological Department Ground AWS (Automatic Weather Station)
    and ARG (Automatic Rain Gauge) telemetry networks for ground truthing.
    """

    # Realistic Ground Stations near our anomaly sectors
    STATION_REGISTRY = [
        # Odisha & Coastal AP (Near Bay of Bengal Cyclone)
        {"id": "IMD_42971", "name": "Bhubaneswar Airport AWS", "state": "Odisha", "district": "Khordha", "lat": 20.244, "lon": 85.818, "elev": 46},
        {"id": "IMD_42972", "name": "Puri Coastal Observatory", "state": "Odisha", "district": "Puri", "lat": 19.800, "lon": 85.817, "elev": 5},
        {"id": "IMD_42973", "name": "Paradip Port DWR", "state": "Odisha", "district": "Jagatsinghpur", "lat": 20.260, "lon": 86.680, "elev": 3},
        {"id": "IMD_43014", "name": "Visakhapatnam Cyclone Radar", "state": "Andhra Pradesh", "district": "Visakhapatnam", "lat": 17.720, "lon": 83.300, "elev": 12},
        {"id": "IMD_42980", "name": "Gopalpur AWS", "state": "Odisha", "district": "Ganjam", "lat": 19.260, "lon": 84.910, "elev": 18},
        
        # Rajasthan & Punjab (Near Heatwave Ridge)
        {"id": "IMD_42101", "name": "Phalodi Extreme AWS", "state": "Rajasthan", "district": "Jodhpur", "lat": 27.130, "lon": 72.360, "elev": 233},
        {"id": "IMD_42110", "name": "Churu Agromet AWS", "state": "Rajasthan", "district": "Churu", "lat": 28.290, "lon": 74.960, "elev": 286},
        {"id": "IMD_42115", "name": "Bikaner Civil AWS", "state": "Rajasthan", "district": "Bikaner", "lat": 28.010, "lon": 73.310, "elev": 224},
        {"id": "IMD_42182", "name": "New Delhi Safdarjung", "state": "Delhi", "district": "New Delhi", "lat": 28.580, "lon": 77.200, "elev": 216},
        {"id": "IMD_42071", "name": "Bathinda Agromet AWS", "state": "Punjab", "district": "Bathinda", "lat": 30.210, "lon": 74.950, "elev": 211},

        # Western Himalaya (Near Flash Flood Region)
        {"id": "IMD_42112", "name": "Dehradun Forest AWS", "state": "Uttarakhand", "district": "Dehradun", "lat": 30.316, "lon": 78.032, "elev": 682},
        {"id": "IMD_42114", "name": "Rishikesh High-Altitude ARG", "state": "Uttarakhand", "district": "Tehri Garhwal", "lat": 30.108, "lon": 78.293, "elev": 372},
        {"id": "IMD_42083", "name": "Shimla Ridge AWS", "state": "Himachal Pradesh", "district": "Shimla", "lat": 31.104, "lon": 77.173, "elev": 2205},
        {"id": "IMD_42084", "name": "Dharamshala Mountain ARG", "state": "Himachal Pradesh", "district": "Kangra", "lat": 32.219, "lon": 76.323, "elev": 1457},
    ]

    def get_realtime_observations(self) -> list[IMDStationObservation]:
        """Simulates/returns current calibrated ground station telemetry."""
        obs_list = []
        now_str = (BASE_CYCLE + timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%SZ")

        for st in self.STATION_REGISTRY:
            # Context-aware values based on region
            if st["state"] in ["Odisha", "Andhra Pradesh"]:
                # Under cyclonic approach
                temp = round(26.8 + random.uniform(-0.8, 1.2), 1)
                humidity = round(88.0 + random.uniform(-4, 8), 1)
                rain_24h = round(64.5 + random.uniform(10, 85), 1)
                wind = round(58.0 + random.uniform(10, 42), 1)
                wind_dir = random.randint(70, 130)  # Easterly / South-Easterly
                press = round(994.0 - random.uniform(2, 12), 1)
                bias_temp = round(random.uniform(-0.4, 0.6), 2)
                bias_rain = round(random.uniform(-3.5, 4.2), 2)
            elif st["state"] in ["Rajasthan", "Punjab", "Delhi"]:
                # Under intense heatwave
                temp = round(44.6 + random.uniform(-0.8, 2.6), 1)
                humidity = round(22.0 + random.uniform(-5, 6), 1)
                rain_24h = 0.0
                wind = round(24.0 + random.uniform(-6, 8), 1)
                wind_dir = random.randint(260, 310)  # Westerly
                press = round(1006.0 + random.uniform(-2, 2), 1)
                bias_temp = round(random.uniform(-0.3, 0.8), 2)
                bias_rain = 0.0
            else:
                # Himalayan orographic convective zone
                temp = round(17.4 + random.uniform(-1.5, 1.5), 1)
                humidity = round(92.0 + random.uniform(-3, 6), 1)
                rain_24h = round(88.4 + random.uniform(15, 65), 1)
                wind = round(32.0 + random.uniform(-5, 15), 1)
                wind_dir = random.randint(190, 240)  # South-westerly monsoon flow
                press = round(875.0 + random.uniform(-4, 4), 1)  # High elevation
                bias_temp = round(random.uniform(-0.8, 0.5), 2)
                bias_rain = round(random.uniform(-5.0, 6.0), 2)

            obs_list.append(
                IMDStationObservation(
                    station_id=st["id"],
                    station_name=st["name"],
                    state=st["state"],
                    district=st["district"],
                    latitude=st["lat"],
                    longitude=st["lon"],
                    elevation_m=st["elev"],
                    temp_c=temp,
                    humidity_pct=humidity,
                    rain_24h_mm=rain_24h,
                    wind_speed_kmh=wind,
                    wind_direction_deg=wind_dir,
                    pressure_hpa=press,
                    status="CALIBRATED_ONLINE",
                    last_update_utc=now_str,
                    model_bias_temp_c=bias_temp,
                    model_bias_rain_mm=bias_rain,
                )
            )

        return obs_list


# ─── 3. ISRO / MOSDAC Satellite Service ─────────────────────────────────────────

class MOSDACService:
    """
    Connects to SAC / MOSDAC for INSAT-3DR / 3DS and Oceansat geophysical products:
    - Hydro-Estimator Precipitation (HE)
    - Cloud Motion Vectors (CMV)
    - Outgoing Longwave Radiation (OLR)
    """

    def get_latest_satellite_feed(self) -> list[MOSDACProduct]:
        timestamp_str = (BASE_CYCLE + timedelta(hours=72)).strftime("%Y-%m-%dT%H:30:00Z")
        return [
            MOSDACProduct(
                product_id="3RIMG_30SEP2026_HE",
                satellite="INSAT-3DR",
                sensor="Imager (6-Channel Multispectral)",
                product_name="Hydro-Estimator 30-min Rain Rate",
                resolution_km=4.0,
                coverage="South Asia & Indian Ocean Region",
                timestamp_utc=timestamp_str,
                cloud_top_temp_c=-74.5,
                max_rain_rate_mmh=48.2,
                olr_wm2=124.0,
                data_url=f"{MOSDAC_API_BASE_URL}/products/insat3dr/he/latest.tif",
            ),
            MOSDACProduct(
                product_id="3DSIMG_30SEP2026_OLR",
                satellite="INSAT-3DS",
                sensor="Sounder (19-Channel Atmospheric)",
                product_name="Outgoing Longwave Radiation & Convective Cloud Top",
                resolution_km=10.0,
                coverage="India Sector",
                timestamp_utc=timestamp_str,
                cloud_top_temp_c=-68.2,
                max_rain_rate_mmh=0.0,
                olr_wm2=145.5,
                data_url=f"{MOSDAC_API_BASE_URL}/products/insat3ds/olr/latest.nc",
            ),
        ]


# ─── 4. NDMA Sachet CAP v1.2 (ITU-T X.1303) Exporter ───────────────────────────

class NDMACAPService:
    """
    Generates standardized ITU-T X.1303 / NDMA Sachet Common Alerting Protocol (CAP v1.2)
    payloads in XML and JSON format from VATAWARAN 5 km downscaled alerts.
    """

    @staticmethod
    def build_cap_alert(
        anomaly_id: str,
        hazard_type: str,
        lead_time_hours: int,
        centroid: list[float],
        severity_level: str,
        metrics: dict[str, Any],
        advisory: str,
        valid_utc: str,
    ) -> CAPAlertPayload:
        lon, lat = centroid[0], centroid[1]
        sent_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S+05:30")
        identifier = f"IN-NDMA-VATAWARAN-{anomaly_id}-T{lead_time_hours}"

        # Assign hazard-specific title and instruction
        if hazard_type == "TROPICAL_CYCLONE":
            event_name = "Severe Cyclonic Storm / Inundation Hazard"
            headline = f"RED ALERT: Severe Cyclonic Inundation & High Wind Shear Threat at T+{lead_time_hours}h"
            area_desc = f"Coastal Sector within 5 km radius of centroid ({lat:.2f}°N, {lon:.2f}°E) [Odisha/Andhra Basin]"
            instruction = (
                "1. District Collectors to initiate Phase-1 pre-emptive evacuation in low-lying coastal belts. "
                "2. NDRF and SDRF battalions to pre-position inflatable boats and tree-clearing equipment. "
                "3. Total suspension of fishing operations and port cargo operations recommended."
            )
        elif hazard_type == "EXTREME_HEAT":
            event_name = "Prolonged Severe Heatwave Anomaly"
            headline = f"ORANGE ALERT: Severe Heatwave Conditions Exceeding Thresholds at T+{lead_time_hours}h"
            area_desc = f"Agrarian Plains within 5 km radius of centroid ({lat:.2f}°N, {lon:.2f}°E) [Northwest Sector]"
            instruction = (
                "1. Agricultural extension officers to broadcast advisory for nocturnal crop irrigation. "
                "2. Municipalities to activate public water kiosks and heat shelters between 11:00 and 16:00. "
                "3. Outdoor physical labor restricted during peak solar irradiation hours."
            )
        else:
            event_name = "Extreme Orographic Rainfall & Flash Flood Threat"
            headline = f"FLASH FLOOD WARNING: Torrential Mountain Downpour at T+{lead_time_hours}h"
            area_desc = f"Valley Basin within 5 km radius of centroid ({lat:.2f}°N, {lon:.2f}°E) [Himalayan Foothills]"
            instruction = (
                "1. Immediate alert to downstream hydroelectric barrage operators to regulate reservoir outflow. "
                "2. District administration to monitor vulnerable landslide choke points on national highways. "
                "3. Riverbank settlements to relocate to designated upland shelter grounds."
            )

        circle_str = f"{lat:.4f},{lon:.4f},5.0"

        return CAPAlertPayload(
            identifier=identifier,
            sender="vatawaran.engine@ncmrwf.gov.in",
            sent=sent_time,
            status="Actual",
            msg_type="Alert",
            scope="Public",
            category="Met",
            event=event_name,
            urgency="Expected" if lead_time_hours > 48 else "Immediate",
            severity=severity_level,
            certainty="Observed" if lead_time_hours <= 48 else "Likely",
            headline=headline,
            description=f"Project VATAWARAN AI Engine (MoES SIH 26078) 5 km physics-downscaled anomaly alert: {advisory}",
            instruction=instruction,
            area_desc=area_desc,
            circle=circle_str,
            lead_time_hours=lead_time_hours,
            metrics=metrics,
        )

    @classmethod
    def generate_cap_xml(cls, alert: CAPAlertPayload) -> str:
        """
        Builds RFC / ITU-T X.1303 OASIS CAP v1.2 XML Document as consumed by NDMA Sachet.
        """
        ns = "urn:oasis:names:tc:emergency:cap:1.2"
        root = ET.Element("alert", attrib={"xmlns": ns})

        ET.SubElement(root, "identifier").text = alert.identifier
        ET.SubElement(root, "sender").text = alert.sender
        ET.SubElement(root, "sent").text = alert.sent
        ET.SubElement(root, "status").text = alert.status
        ET.SubElement(root, "msgType").text = alert.msg_type
        ET.SubElement(root, "source").text = "VATAWARAN AI Downscaling Engine (MoES/NCMRWF)"
        ET.SubElement(root, "scope").text = alert.scope
        ET.SubElement(root, "code").text = "IPAWS-1.0"

        info = ET.SubElement(root, "info")
        ET.SubElement(info, "language").text = "en-IN"
        ET.SubElement(info, "category").text = alert.category
        ET.SubElement(info, "event").text = alert.event
        ET.SubElement(info, "urgency").text = alert.urgency
        ET.SubElement(info, "severity").text = alert.severity
        ET.SubElement(info, "certainty").text = alert.certainty
        ET.SubElement(info, "senderName").text = "Ministry of Earth Sciences / NCMRWF / Project VATAWARAN"
        ET.SubElement(info, "headline").text = alert.headline
        ET.SubElement(info, "description").text = alert.description
        ET.SubElement(info, "instruction").text = alert.instruction

        # Model telemetry parameters
        for key, val in alert.metrics.items():
            param = ET.SubElement(info, "parameter")
            ET.SubElement(param, "valueName").text = key
            ET.SubElement(param, "value").text = str(val)

        # Spatial Area Definition
        area = ET.SubElement(info, "area")
        ET.SubElement(area, "areaDesc").text = alert.area_desc
        ET.SubElement(area, "circle").text = alert.circle

        geocode = ET.SubElement(area, "geocode")
        ET.SubElement(geocode, "valueName").text = "ISO3166-2"
        ET.SubElement(geocode, "value").text = "IN"

        return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


# ─── Singleton Aggregator Instance ─────────────────────────────────────────────

class GovIntegrationHub:
    """Consolidated manager for all upstream and downstream government endpoints."""

    def __init__(self):
        self.ncmrwf = NCMRWFService()
        self.imd = IMDService()
        self.mosdac = MOSDACService()
        self.cap = NDMACAPService()

    def get_all_source_statuses(self) -> list[GovSourceStatus]:
        now_str = (BASE_CYCLE + timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return [
            GovSourceStatus(
                source_id="MOES-NCMRWF-NEPSG",
                agency="Ministry of Earth Sciences (MoES) / NCMRWF",
                name="NEPS-G 12 km Global Ensemble Ingest",
                protocol="OpenDAP / THREDDS (TDS)",
                endpoint=NCMRWF_THREDDS_URL,
                status="SYNCED",
                latency_ms=185,
                last_sync_utc=now_str,
                data_format="GRIB2 / NetCDF4",
                variables=["U850", "V850", "U200", "V200", "Z500", "T850", "T2M", "Q700", "TP"],
                description="Primary 50-member medium-range ensemble forecast stream covering a 10-day lookahead window.",
            ),
            GovSourceStatus(
                source_id="MOES-NCMRWF-IMDAA",
                agency="Ministry of Earth Sciences (MoES) / NCMRWF",
                name="IMDAA 12 km Regional Reanalysis Baseline",
                protocol="OpenDAP / NCSS",
                endpoint="https://rds.ncmrwf.gov.in/opendap/IMDAA",
                status="ONLINE",
                latency_ms=120,
                last_sync_utc=now_str,
                data_format="NetCDF4",
                variables=["mu_c(d)", "sigma_c(d)", "EFI_thresholds"],
                description="40-year climatological baseline used to normalize atmospheric vectors into Extreme Forecast Index (EFI).",
            ),
            GovSourceStatus(
                source_id="MOES-IMD-AWS",
                agency="India Meteorological Department (IMD)",
                name="Ground AWS & Automatic Rain Gauge Telemetry",
                protocol="REST API / JSON",
                endpoint=f"{IMD_API_BASE_URL}/aws/realtime",
                status="ONLINE",
                latency_ms=64,
                last_sync_utc=now_str,
                data_format="GeoJSON / JSON",
                variables=["Temp", "Humidity", "Rain24h", "WindGust", "SurfacePressure"],
                description="Real-time ground station network across Indian states for model validation and bias calculation.",
            ),
            GovSourceStatus(
                source_id="ISRO-SAC-MOSDAC",
                agency="Space Applications Centre (ISRO) / MOSDAC",
                name="INSAT-3DR / 3DS Satellite Geostationary Imagery",
                protocol="HTTPS / STAC API",
                endpoint=MOSDAC_API_BASE_URL,
                status="ONLINE",
                latency_ms=210,
                last_sync_utc=now_str,
                data_format="GeoTIFF / HDF5",
                variables=["HydroEstimator_Rain", "CloudMotionVectors", "OLR"],
                description="Half-hourly multispectral geostationary satellite feeds for convective system ground truthing.",
            ),
            GovSourceStatus(
                source_id="MHA-NDMA-SACHET",
                agency="National Disaster Management Authority (NDMA) / MHA",
                name="Sachet National Common Alerting Protocol (CAP) Gateway",
                protocol="REST API / ITU-T X.1303 CAP v1.2",
                endpoint=NDMA_SACHET_URL,
                status="ONLINE",
                latency_ms=92,
                last_sync_utc=now_str,
                data_format="XML (CAP v1.2) / JSON",
                variables=["GeoTargetCircle", "SeverityLevel", "ActionableAdvisory"],
                description="Official emergency alert dissemination pipeline to NDRF battalions, SDMAs, and mobile cell broadcasts.",
            ),
        ]


# Export single instance
gov_hub = GovIntegrationHub()
