# VATAWARAN 🌩️

> **AI-Driven Spatio-Temporal Extreme Weather Tracking & Downscaling Engine**  
> Smart India Hackathon 2026 · Problem Statement ID: **26078**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)
[![SIH 2026](https://img.shields.io/badge/SIH-2026-orange.svg)](https://www.sih.gov.in)

---

## 📌 Problem Statement

Modern operational NWP models (NCMRWF NEPS-G at 12 km, NCUM deterministic) produce medium-range ensemble forecasts with wide spread in chaotic atmospheric regimes. Meteorologists must manually scan multi-terabyte 4D datasets to detect extreme-weather anomalies and downscale them for district-level impact assessment.

**VATAWARAN** automates this pipeline with a two-stage hybrid AI architecture:

| Stage | Technology | Output |
|---|---|---|
| **Stage 1 — GNN Tracking** | Spherical Graph Neural Network on icosahedral mesh | Anomaly trajectories, EFI scores, bounding boxes |
| **Stage 2 — Diffusion Downscaling** | Physics-constrained DDPM (DDIM scheduler) | 5 km downscaled precipitation, wind, temperature fields |
| **Delivery Layer** | FastAPI + MapLibre GL GIS dashboard | GeoJSON alerts, interactive synoptic chart |

---

## 🗂️ Project Structure

```
VATAWARAN/
├── backend/
│   ├── __init__.py
│   └── main.py              # FastAPI server — all API endpoints
├── frontend/
│   └── index.html           # Interactive GIS dashboard (zero dependencies)
├── requirements.txt         # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/VATAWARAN.git
cd VATAWARAN
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the server
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open the dashboard
Visit **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/system/status` | Pipeline health, GPU info, forecast cycle |
| `GET` | `/api/v1/forecast/track` | Anomaly trajectories with EFI scores |
| `POST` | `/api/v1/downscale/generate` | Run 5 km diffusion downscaling |

### Example — Track Anomalies
```bash
curl "http://localhost:8000/api/v1/forecast/track?lead_time_min=48&lead_time_max=240&min_efi=0.80"
```

### Example — Run Downscaling
```bash
curl -X POST http://localhost:8000/api/v1/downscale/generate \
  -H "Content-Type: application/json" \
  -d '{
    "anomaly_id": "BOB-CYCLONE-2026-01",
    "lead_time_hours": 96,
    "target_variables": ["TP", "U10M", "V10M", "T2M"],
    "num_ensemble_samples": 10
  }'
```

---

## 🌐 Frontend Features

- **Synoptic chart** — MapLibre GL with trajectory tracks, bounding boxes, centroid markers
- **Anomaly dossiers** — EFI score, confidence, lead-time window per anomaly
- **Downscaling animation** — Real-time 50-step DDIM denoising progress with segmented bar
- **5 km heatmap overlay** — Precipitation / heat field rendered on map post-downscale
- **Operational advisory** — Auto-generated alert text with severity classification

---

## 🏗️ Architecture

```
NEPS-G Ensemble (12 km)
        │
        ▼
 ┌─────────────┐
 │  Stage 1    │  Spherical GNN  →  Anomaly trajectories + EFI
 │  GNN Track  │  Icosahedral mesh
 └──────┬──────┘
        │ Crop + Condition
        ▼
 ┌─────────────┐
 │  Stage 2    │  Physics-DDPM   →  5 km downscaled fields
 │  Diffusion  │  DDIM · 50 steps
 └──────┬──────┘
        │ GeoJSON
        ▼
 FastAPI Delivery Layer  →  MapLibre GL Dashboard
```

---

## 📦 Dependencies

```
fastapi          — REST API framework
uvicorn          — ASGI server
pydantic         — Data validation & serialisation
numpy            — Numerical array ops
scipy            — Spatial interpolation (prototype simulation)
```

---

## 🗺️ Roadmap (Post-Prototype)

- [ ] Real GRIB2/NetCDF4 ingestion (xarray + cfgrib)
- [ ] Spherical GNN on icosahedral mesh (PyTorch Geometric)
- [ ] Train conditional DDPM with physics loss (HuggingFace Diffusers)
- [ ] ERA5 30-year climatological baseline for real EFI
- [ ] SRTM DEM conditioning for diffusion model
- [ ] OAuth2 / API key authentication
- [ ] Cloud Optimized GeoTIFF raster export
- [ ] Kubernetes Helm chart for production deployment

---

## 👥 Team

Built for **Smart India Hackathon 2026** · Problem Statement #26078  
Organized by **Ministry of Education, Government of India**

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
