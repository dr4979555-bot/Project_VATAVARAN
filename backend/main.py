"""
VATAWARAN API Server — Working Prototype
=========================================
Implements the core endpoints from the Technical Requirements Document:
  GET  /api/v1/forecast/track       — Anomaly trajectory retrieval
  POST /api/v1/downscale/generate   — Physics-constrained diffusion downscaling
  GET  /api/v1/system/status        — Pipeline health & stage indicators
"""

import math
import os
import random
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ─── App Init ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VATAWARAN API",
    description=(
        "AI-Driven Spatio-Temporal Extreme Weather Tracking & Downscaling Engine. "
        "Smart India Hackathon 2026 | Problem Statement ID: 26078"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pydantic Models (TRD §5 Contract) ────────────────────────────────────────


class TrajectoryPoint(BaseModel):
    lead_time_hours: int
    valid_utc: str
    centroid: list[float]
    bounding_box: list[float]  # [lat_min, lon_min, lat_max, lon_max]
    efi_score: float


class AnomalyData(BaseModel):
    anomaly_id: str
    hazard_type: str
    confidence_score: float
    peak_efi: float
    description: str
    trajectories: list[TrajectoryPoint]


class TrackResponse(BaseModel):
    status: str
    forecast_cycle: str
    anomalies_detected: int
    data: list[AnomalyData]


class DownscaleRequest(BaseModel):
    anomaly_id: str
    lead_time_hours: int = 96
    target_variables: list[str] = ["TP", "U10M", "V10M"]
    num_ensemble_samples: int = 10


class PipelineStage(BaseModel):
    stage: int
    name: str
    status: str  # "completed" | "running" | "idle"
    latency_ms: Optional[int] = None


class SystemStatus(BaseModel):
    status: str
    uptime_seconds: int
    gpu_available: bool
    gpu_name: str
    forecast_cycle: str
    pipeline_stages: list[PipelineStage]


# ─── Mock Data ─────────────────────────────────────────────────────────────────

BASE_TIME = datetime(2026, 9, 30, 0, 0, 0)

MOCK_ANOMALIES: list[AnomalyData] = [
    # ── Tropical Cyclone in Bay of Bengal ──
    AnomalyData(
        anomaly_id="BOB-CYCLONE-2026-01",
        hazard_type="TROPICAL_CYCLONE",
        confidence_score=0.94,
        peak_efi=0.96,
        description=(
            "Intense tropical cyclonic disturbance detected over central Bay of Bengal. "
            "Projected northwestward track toward Odisha-Andhra coast with peak sustained "
            "winds exceeding 140 km/h and extreme precipitation bands."
        ),
        trajectories=[
            TrajectoryPoint(
                lead_time_hours=48,
                valid_utc=(BASE_TIME + timedelta(hours=48)).isoformat() + "Z",
                centroid=[88.90, 15.20],
                bounding_box=[13.0, 86.5, 17.5, 91.5],
                efi_score=0.82,
            ),
            TrajectoryPoint(
                lead_time_hours=72,
                valid_utc=(BASE_TIME + timedelta(hours=72)).isoformat() + "Z",
                centroid=[88.25, 17.50],
                bounding_box=[15.0, 85.5, 20.0, 91.0],
                efi_score=0.91,
            ),
            TrajectoryPoint(
                lead_time_hours=96,
                valid_utc=(BASE_TIME + timedelta(hours=96)).isoformat() + "Z",
                centroid=[87.80, 19.80],
                bounding_box=[17.5, 85.0, 22.0, 90.5],
                efi_score=0.94,
            ),
            TrajectoryPoint(
                lead_time_hours=120,
                valid_utc=(BASE_TIME + timedelta(hours=120)).isoformat() + "Z",
                centroid=[86.90, 21.20],
                bounding_box=[19.0, 84.0, 23.5, 89.5],
                efi_score=0.96,
            ),
            TrajectoryPoint(
                lead_time_hours=144,
                valid_utc=(BASE_TIME + timedelta(hours=144)).isoformat() + "Z",
                centroid=[85.50, 22.80],
                bounding_box=[20.5, 83.0, 25.0, 88.0],
                efi_score=0.88,
            ),
        ],
    ),
    # ── Northwest India Heatwave ──
    AnomalyData(
        anomaly_id="NWI-HEATWAVE-2026-03",
        hazard_type="EXTREME_HEAT",
        confidence_score=0.89,
        peak_efi=0.91,
        description=(
            "Persistent 850 hPa geopotential ridge over Rajasthan-Punjab belt. "
            "Surface temperatures projected to exceed 46°C for 72+ consecutive hours. "
            "Significant agricultural and public health hazard."
        ),
        trajectories=[
            TrajectoryPoint(
                lead_time_hours=72,
                valid_utc=(BASE_TIME + timedelta(hours=72)).isoformat() + "Z",
                centroid=[73.50, 27.80],
                bounding_box=[26.0, 71.0, 30.0, 76.0],
                efi_score=0.84,
            ),
            TrajectoryPoint(
                lead_time_hours=96,
                valid_utc=(BASE_TIME + timedelta(hours=96)).isoformat() + "Z",
                centroid=[74.20, 28.30],
                bounding_box=[26.5, 71.5, 30.5, 77.0],
                efi_score=0.89,
            ),
            TrajectoryPoint(
                lead_time_hours=120,
                valid_utc=(BASE_TIME + timedelta(hours=120)).isoformat() + "Z",
                centroid=[75.00, 28.50],
                bounding_box=[26.5, 72.0, 31.0, 78.0],
                efi_score=0.91,
            ),
            TrajectoryPoint(
                lead_time_hours=144,
                valid_utc=(BASE_TIME + timedelta(hours=144)).isoformat() + "Z",
                centroid=[75.50, 28.20],
                bounding_box=[26.0, 72.5, 30.5, 78.5],
                efi_score=0.87,
            ),
        ],
    ),
    # ── Western Himalayan Extreme Rainfall ──
    AnomalyData(
        anomaly_id="WHR-RAIN-2026-07",
        hazard_type="EXTREME_PRECIPITATION",
        confidence_score=0.86,
        peak_efi=0.88,
        description=(
            "Orographically enhanced extreme precipitation event over Uttarakhand-Himachal "
            "foothills. Moisture-laden westerly troughs interacting with steep terrain. "
            "Flash flood and landslide risk critical in narrow river valleys."
        ),
        trajectories=[
            TrajectoryPoint(
                lead_time_hours=48,
                valid_utc=(BASE_TIME + timedelta(hours=48)).isoformat() + "Z",
                centroid=[79.50, 30.50],
                bounding_box=[29.5, 78.0, 31.5, 81.0],
                efi_score=0.80,
            ),
            TrajectoryPoint(
                lead_time_hours=72,
                valid_utc=(BASE_TIME + timedelta(hours=72)).isoformat() + "Z",
                centroid=[79.20, 31.00],
                bounding_box=[29.8, 77.5, 32.0, 81.0],
                efi_score=0.86,
            ),
            TrajectoryPoint(
                lead_time_hours=96,
                valid_utc=(BASE_TIME + timedelta(hours=96)).isoformat() + "Z",
                centroid=[78.80, 30.80],
                bounding_box=[29.5, 77.0, 32.2, 80.5],
                efi_score=0.88,
            ),
            TrajectoryPoint(
                lead_time_hours=120,
                valid_utc=(BASE_TIME + timedelta(hours=120)).isoformat() + "Z",
                centroid=[78.30, 30.20],
                bounding_box=[28.8, 76.5, 31.5, 80.0],
                efi_score=0.82,
            ),
        ],
    ),
]

# Map anomaly IDs to their data for quick lookup
ANOMALY_MAP = {a.anomaly_id: a for a in MOCK_ANOMALIES}


# ─── Synthetic Downscale Grid Generator ────────────────────────────────────────


def _generate_downscale_grid(
    center_lon: float,
    center_lat: float,
    hazard_type: str,
    grid_extent_km: float = 200,
    resolution_km: float = 5,
) -> list[dict]:
    """
    Generate a synthetic 5 km downscaled grid around an anomaly centroid.
    Uses 2D Gaussian + asymmetric perturbation to mimic realistic fields.
    """
    n = int(grid_extent_km / resolution_km)
    deg_step = resolution_km / 111.0
    half = n // 2
    features = []

    # Slight directional bias to break symmetry (realistic)
    bias_x = random.uniform(-0.15, 0.15)
    bias_y = random.uniform(-0.10, 0.10)
    # Random rotation for asymmetry
    rot = random.uniform(0, 2 * math.pi)

    for i in range(-half, half + 1, 2):  # step=2 to reduce point count
        for j in range(-half, half + 1, 2):
            lon = center_lon + j * deg_step
            lat = center_lat + i * deg_step

            # Rotated / biased distance
            dx = j + bias_x * half
            dy = i + bias_y * half
            rx = dx * math.cos(rot) - dy * math.sin(rot)
            ry = dx * math.sin(rot) + dy * math.cos(rot)
            dist = math.sqrt(rx ** 2 + ry ** 2)
            sigma = half * 0.38

            base = math.exp(-0.5 * (dist / sigma) ** 2)
            noise = random.gauss(0, 0.04)

            if hazard_type == "TROPICAL_CYCLONE":
                precip_p50 = max(0, round(280 * base + noise * 40, 1))
                precip_p90 = round(precip_p50 * 1.38, 1)
                precip_p99 = round(precip_p50 * 1.68, 1)
                # Eye-wall wind pattern (peak slightly off-center)
                wind_r = math.sqrt((dx - 2) ** 2 + (dy + 1) ** 2)
                wind_base = math.exp(-0.5 * ((wind_r - 4) / (half * 0.25)) ** 2)
                wind_gust = max(0, round(155 * wind_base + random.gauss(0, 8), 1))
                temp = round(26.5 + 3 * base + random.gauss(0, 0.4), 1)
            elif hazard_type == "EXTREME_HEAT":
                precip_p50 = 0.0
                precip_p90 = 0.0
                precip_p99 = round(max(0, random.gauss(2, 3)), 1)
                wind_gust = round(max(0, 25 + random.gauss(0, 8)), 1)
                temp = round(38 + 8 * base + random.gauss(0, 0.6), 1)
            else:  # EXTREME_PRECIPITATION
                precip_p50 = max(0, round(220 * base + noise * 35, 1))
                precip_p90 = round(precip_p50 * 1.42, 1)
                precip_p99 = round(precip_p50 * 1.75, 1)
                wind_gust = max(0, round(65 * base + random.gauss(0, 6), 1))
                temp = round(18 + 4 * base + random.gauss(0, 0.5), 1)

            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(lon, 4), round(lat, 4)],
                    },
                    "properties": {
                        "precip_p50_mm": precip_p50,
                        "precip_p90_mm": precip_p90,
                        "precip_p99_mm": precip_p99,
                        "wind_gust_kmh": wind_gust,
                        "temperature_c": temp,
                    },
                }
            )

    return features


# ─── API Endpoints ─────────────────────────────────────────────────────────────


@app.get("/api/v1/system/status", response_model=SystemStatus)
async def system_status():
    """Returns current pipeline health and stage indicators."""
    return SystemStatus(
        status="operational",
        uptime_seconds=86400,
        gpu_available=True,
        gpu_name="NVIDIA A10G (24 GB VRAM)",
        forecast_cycle=BASE_TIME.isoformat() + "Z",
        pipeline_stages=[
            PipelineStage(stage=1, name="Data Ingestion & Normalization", status="completed", latency_ms=2340),
            PipelineStage(stage=2, name="Spherical GNN Anomaly Tracker", status="completed", latency_ms=7820),
            PipelineStage(stage=3, name="Dynamic Spatial Cropping", status="completed", latency_ms=410),
            PipelineStage(stage=4, name="Physics-Guided Diffusion Downscaling", status="idle", latency_ms=None),
            PipelineStage(stage=5, name="Validation & Alert Delivery", status="idle", latency_ms=None),
        ],
    )


@app.get("/api/v1/forecast/track", response_model=TrackResponse)
async def get_forecast_tracks(
    lead_time_min: int = Query(default=48, description="Minimum lead time in hours"),
    lead_time_max: int = Query(default=240, description="Maximum lead time in hours"),
    min_efi: float = Query(default=0.80, description="Minimum EFI threshold"),
):
    """
    TRD §5.1 — Retrieves spatio-temporal trajectories of detected extreme anomalies.
    Filters by lead time window and minimum EFI score.
    """
    filtered = []
    for anomaly in MOCK_ANOMALIES:
        if anomaly.peak_efi < min_efi:
            continue
        traj = [
            t
            for t in anomaly.trajectories
            if lead_time_min <= t.lead_time_hours <= lead_time_max
               and t.efi_score >= min_efi
        ]
        if traj:
            filtered.append(
                AnomalyData(
                    anomaly_id=anomaly.anomaly_id,
                    hazard_type=anomaly.hazard_type,
                    confidence_score=anomaly.confidence_score,
                    peak_efi=anomaly.peak_efi,
                    description=anomaly.description,
                    trajectories=traj,
                )
            )

    return TrackResponse(
        status="success",
        forecast_cycle=BASE_TIME.isoformat() + "Z",
        anomalies_detected=len(filtered),
        data=filtered,
    )


@app.post("/api/v1/downscale/generate")
async def generate_downscale(req: DownscaleRequest):
    """
    TRD §5.2 — Executes Stage 2 physics-constrained diffusion downscaling.
    Returns a GeoJSON FeatureCollection with the 5 km grid and max-impact alert.
    """
    anomaly = ANOMALY_MAP.get(req.anomaly_id)
    if anomaly is None:
        return {"status": "error", "message": f"Unknown anomaly_id: {req.anomaly_id}"}

    # Find the trajectory point closest to the requested lead time
    best = min(anomaly.trajectories, key=lambda t: abs(t.lead_time_hours - req.lead_time_hours))
    center_lon, center_lat = best.centroid

    # Generate synthetic 5 km downscaled grid
    grid_features = _generate_downscale_grid(center_lon, center_lat, anomaly.hazard_type)

    # Find peak values across grid for the alert feature
    peak_precip_p50 = max((f["properties"]["precip_p50_mm"] for f in grid_features), default=0)
    peak_precip_p90 = max((f["properties"]["precip_p90_mm"] for f in grid_features), default=0)
    peak_precip_p99 = max((f["properties"]["precip_p99_mm"] for f in grid_features), default=0)
    peak_wind = max((f["properties"]["wind_gust_kmh"] for f in grid_features), default=0)

    # Physics QA simulation (always passes in prototype)
    mass_residual = round(random.uniform(0.005, 0.035), 4)
    moist_residual = round(random.uniform(0.02, 0.10), 4)
    physics_passed = mass_residual <= 0.05 and moist_residual <= 0.15

    # Severity determination
    if anomaly.hazard_type == "TROPICAL_CYCLONE":
        severity = "EXTREME" if peak_wind > 120 else "SEVERE"
        advisory = (
            "Extreme localized inundation and destructive wind shear expected. "
            "Immediate evacuation advisory for coastal settlements within 5 km impact radius."
        )
    elif anomaly.hazard_type == "EXTREME_HEAT":
        severity = "SEVERE"
        advisory = (
            "Prolonged extreme heat exceeding 44°C for 72+ hours. "
            "Critical risk to agriculture and vulnerable populations. Emergency irrigation advised."
        )
    else:
        severity = "SEVERE" if peak_precip_p99 > 200 else "HIGH"
        advisory = (
            "High likelihood of flash flooding in steep terrain. "
            "Orographic enhancement producing extreme localized precipitation bands."
        )

    # Max-impact alert feature (the centroid with 5 km radius)
    alert_feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [center_lon, center_lat]},
        "properties": {
            "alert_type": "SEVERE_WEATHER_IMPACT",
            "impact_radius_km": 5.0,
            "severity_level": severity,
            "metrics": {
                "peak_precipitation_p50_mm": round(peak_precip_p50, 1),
                "peak_precipitation_p90_mm": round(peak_precip_p90, 1),
                "peak_precipitation_p99_mm": round(peak_precip_p99, 1),
                "peak_wind_gust_kmh": round(peak_wind, 1),
            },
            "advisory": advisory,
        },
    }

    return {
        "type": "FeatureCollection",
        "metadata": {
            "anomaly_id": req.anomaly_id,
            "hazard_type": anomaly.hazard_type,
            "lead_time_hours": best.lead_time_hours,
            "valid_utc": best.valid_utc,
            "resolution_km": 5.0,
            "num_ensemble_samples": req.num_ensemble_samples,
            "physics_qa_passed": physics_passed,
            "physics_residuals": {
                "mass_divergence": mass_residual,
                "moisture_flux": moist_residual,
            },
            "degraded_mode": not physics_passed,
            "diffusion_steps": 50,
            "scheduler": "DDIM",
        },
        "features": [alert_feature] + grid_features,
    }


# ─── Serve Frontend ───────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<h1>VATAWARAN API is running. Frontend not found.</h1>")


# Mount frontend static assets (CSS, JS, images if any)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
