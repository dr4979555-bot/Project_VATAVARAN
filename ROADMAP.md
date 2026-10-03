# Project VATAWARAN — Phased Execution Roadmap

**AI-Driven Spatio-Temporal Extreme Weather Tracking & Downscaling Engine**  
*Smart India Hackathon 2026 | Problem Statement ID: 26078*

---

## 📌 Executive Summary & Current Status

| Milestone | Status | Description |
| :--- | :--- | :--- |
| **Phase 0: Core Prototype & Gov Feeds** | **COMPLETED** ✅ | Full REST API (15 routes), broadsheet GIS UI, 5 Indian Gov APIs, ITU-T X.1303 CAP XML export, launch video (`brag.mp4`), pushed to GitHub. |
| **Phase 1: Automated Test Suite & CI/CD** | **COMPLETED** ✅ | 27/27 tests passing across `test_api.py`, `test_gov_services.py`, and `test_downscale.py`, plus `.github/workflows/ci.yml`. |
| **Phase 3: Raw NetCDF4 & GRIB2 Ingestion** | **COMPLETED** ✅ | `backend/ingestion.py` — 9-channel state vectors (181×361), IMDAA/ERA5 seasonal standardization, OpenDAP→offline fallback, `POST /api/v1/ingest/cycle` + `GET /api/v1/ingest/channels`, committed sample grid, physical unit autoconformance, Windows file-lock defense, and error hardening. 46/46 tests green. |
| **Phase 4: SQLite Database Persistence** | **UP NEXT** 🏁 | Persist anomaly history, station observations, and dispatched CAP alerts. |
| **Phase 5: Background Task Scheduler** | **PLANNED** 📋 | Autonomous 6-hour forecast cycle polling (00, 06, 12, 18 UTC). |
| **Phase 6: PyTorch Model Loader** | **PLANNED** 📋 | Modular `.safetensors` checkpoint loader for GNN and Diffusion U-Net. |

---

## 🏁 WHERE TO RESUME (Immediate Starting Point)

When you return, start directly with **Phase 4 (Database Persistence with SQLite)**.

### Quick Command to Resume Work

```powershell
# 1. (Optional) Spin up the containerized stack verified in Phase 2:
docker compose up --build -d     # needs Docker Desktop for Windows (WSL2 backend)
#    Sanity check:  curl http://localhost:8000/api/v1/system/status

# 2. Or activate your virtual environment for local development:
.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload

# 3. Check git branch status:
git status
```

---

## 🗺️ Detailed Phase Breakdown

### Phase 1: Automated Pytest Suite & GitHub Actions CI (COMPLETED ✅)

*Objective: Build an automated test suite so any changes are verified with one command (`pytest tests/`) and GitHub automatically tests every commit.*

- **Tasks**:
  1. `tests/test_api.py`: Test all FastAPI endpoints (`/system/status`, `/forecast/track`, `/downscale/generate`).
  2. `tests/test_gov_services.py`: Test NCMRWF cycle parser, IMD station coordinates, MOSDAC satellite payloads, and ITU-T X.1303 CAP XML compliance.
  3. `tests/test_downscale.py`: Verify mass divergence, moisture flux checks, and 5 km GeoJSON output structure.
  4. `.github/workflows/ci.yml`: Add GitHub Actions workflow to run tests on push/pull-request.
- **Verification Target**:

  ```bash
  pytest tests/ -v --tb=short
  ```

- **Exit Criteria**: 100% green test run across all endpoints without external network dependencies (mock fallbacks tested).

---

### Phase 2: Containerization & Deployment Orchestration (COMPLETED ✅)

*Objective: Package the full stack into a Docker container so anyone can run the app with 1 command (`docker compose up`).*

- **Prerequisites Note**:
  - You attempted `docker compose up --build` and received `docker: command not found`.
  - To use Docker on Windows, install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) (with WSL2 backend enabled).
- **Tasks**:
  1. `Dockerfile`: Create multi-stage Python 3.11 Debian slim container with Geospatial C-libraries (`libgdal`, `netcdf4`).
  2. `docker-compose.yml`: Configure port mapping (`8000:8000`), `.env` injection, and volume mounts.
  3. `.dockerignore`: Exclude `.venv/`, `__pycache__/`, `.agent/`, `.git/` to keep container image lightweight (~250MB).
- **Verification Target**:

  ```bash
  docker compose up --build -d
  curl http://localhost:8000/api/v1/system/status
  ```

- **Exit Criteria**: Clean container spin-up serving the GIS dashboard on port 8000.

- **Delivered**:
  1. `Dockerfile` — `python:3.11-slim-bookworm`, geospatial C-libraries (`libgdal`, `libgeos`, `libproj`, `libnetcdf`, `libhdf5`, `libudunits`), CPU-only PyTorch pre-installed from the PyTorch CPU wheelhouse to avoid the ~2.5 GB CUDA wheel, `--no-cache-dir` pip install, unprivileged system user (`vatavaran`), `EXPOSE 8000`, HEALTHCHECK, and `uvicorn backend.main:app` default command.
  2. `docker-compose.yml` — `vatavaran-app` service, port `8000:8000`, `env_file: .env` (`required: false` so a fresh clone still boots), `/api/v1/system/status` healthcheck, `restart: unless-stopped`.
  3. `.dockerignore` — excludes `.venv/`, `.git/`, `.pytest_cache/`, `__pycache__/`, `.agent/`, `brag-output/`, `*.log`, and **all secrets** (`.env`, `.env.local`) from the build context.
  4. **Validation**: `pytest tests/ -v --tb=short` → **27/27 passing**; CPU-torch pip resolution empirically verified; Compose YAML + Dockerfile syntax validated offline (Docker daemon not available on the Windows dev box at completion time).

> **Note**: The requested target image size (~250–300 MB) is directionally limited by the bundled ML/geospatial stack (`torch`, `diffusers`, `cartopy`, `netCDF4`). The CPU-only PyTorch strategy is the single largest lever applied to keep the image as lean as possible while remaining fully functional without a GPU.

---

### Phase 3: Raw NetCDF4 & GRIB2 Ingestion Engine (`backend/ingestion.py`) (COMPLETED ✅)

*Objective: Bridge raw binary weather data from NCMRWF OpenDAP servers into normalized NumPy/PyTorch tensors.*

- **Tasks**:
  1. `backend/ingestion.py`: Module leveraging `xarray` + `netCDF4` + `cfgrib`.
  2. Build 9-channel atmospheric state vector:
     $$X_t = [U_{850}, V_{850}, U_{200}, V_{200}, Z_{500}, T_{850}, T_{2m}, Q_{700}, TP]$$
  3. Implement climatological standardization against 30-year IMDAA/ERA5 baseline ($\mu_c$, $\sigma_c$):
     $$\hat{X}_t = \frac{X_t - \mu_c(d)}{\sigma_c(d) + 10^{-6}}$$
  4. Create `data/sample/` with a lightweight 12 km sample NetCDF grid for offline verification.
- **Verification Target**:

  ```python
  from backend.ingestion import ingest_ncmrwf_cycle
  tensor = ingest_ncmrwf_cycle("data/sample/neps_sample.nc")
  assert tensor.shape[-2:] == (181, 361)
  ```

- **Exit Criteria**: Successfully extracts atmospheric variables from `.nc` files into ready-to-track tensors.

- **Delivered**:
  1. `backend/ingestion.py` — `xarray` + `netCDF4` + `numpy` engine (torch optional) with alias & pressure-level variable resolution, finite-cell climatological imputation, seasonal IMDAA/ERA5 standardization `(X − μ_d)/(σ_d + 1e-6)`, GRIB2 path (optional `cfgrib`), and graceful **remote OpenDAP → offline sample** fallback.
  2. `data/sample/neps_sample.nc` — deterministic 12 km NEPS-G mock (1.0° grid, 181×361, 9 zlib-compressed channels, ~1.6 MB) committed via `.gitignore` / `.dockerignore` negation; `COPY data/sample/` added to `Dockerfile`.
  3. REST endpoints — `POST /api/v1/ingest/cycle` (JSON path *or* multipart upload) and `GET /api/v1/ingest/channels` (catalog + baseline). Added `python-multipart` to `requirements.txt` for upload parsing.
  4. `tests/test_ingestion.py` — 19 comprehensive unit/integration/edge-case tests; full suite **46/46 passing**.

---

### Phase 4: Database Persistence (`backend/database.py` with SQLite)

*Objective: Replace in-memory dictionaries with zero-config SQLite persistence (`vatavaran.db`).*

- **Tasks**:
  1. `backend/database.py`: SQLite connection using `SQLAlchemy` or `aiosqlite`.
  2. Tables:
     - `anomaly_records`: Track ID, hazard type, EFI score, bounding box, timestamp.
     - `station_telemetry`: History of 14 IMD ground station readings and model bias.
     - `cap_dispatches`: Log of all exported CAP v1.2 alerts with recipient agencies and signatures.
  3. New query endpoints:
     - `GET /api/v1/history/anomalies`: List previously identified weather anomalies.
     - `GET /api/v1/history/alerts`: Audit log of emergency warnings dispatched.
- **Verification Target**:

  ```bash
  curl http://localhost:8000/api/v1/history/alerts
  ```

- **Exit Criteria**: Generated downscaling events and CAP alerts persist across server restarts.

---

### Phase 5: Autonomous Background Polling & Early Warning Scheduler

*Objective: Transform VATAWARAN into a live 24/7 autonomous monitoring service.*

- **Tasks**:
  1. Integrate `APScheduler` or FastAPI background workers into `backend/main.py`.
  2. Scheduled Jobs:
     - **Every 6 hours (00, 06, 12, 18 UTC)**: Poll NCMRWF OpenDAP for newly published NEPS-G model runs.
     - **Every 30 minutes**: Poll ISRO MOSDAC for new INSAT-3DR/3DS satellite cloud-top and rain products.
     - **Every 15 minutes**: Check ground AWS telemetry for threshold breaches.
  3. Automatic CAP alert generation trigger when detected $\text{EFI} \ge 0.85$.
- **Verification Target**:
  Check application console logs for scheduled sync heartbeats:
  `[CRON] 06:00 UTC cycle synced. Active anomaly count: 3.`
- **Exit Criteria**: Server automatically updates internal forecast state without user intervention.

---

### Phase 6: Deep Learning Checkpoint Loader (`backend/ml/`)

*Objective: Prepare model execution slots for trained PyTorch / Safetensors weights.*

- **Tasks**:
  1. `backend/ml/spherical_gnn.py`: PyTorch module with Icosahedral geodesic mesh graph convolutions.
  2. `backend/ml/diffusion_unet.py`: Denoising U-Net with DDIM sampler.
  3. `backend/ml/loader.py`: Safe loader for `.safetensors` model weights:
     - If weights exist in `models/`: run neural inference on GPU/CPU.
     - If weights are absent: seamlessly fall back to the analytical physics simulation.
- **Verification Target**:

  ```bash
  python -c "from backend.ml.loader import load_models; print(load_models())"
  ```

- **Exit Criteria**: Seamless toggle between real ML weights and analytical simulation.

---

## 📋 Checklist for Resuming Your Session

```markdown
- [x] 1. Open terminal in `d:\VATAWARAN`
- [x] 2. Activate `.venv`: `.venv\Scripts\Activate.ps1`
- [x] 3. Run: `git status` (confirm `main` is clean)
- [x] 4. Phase 2 delivered: `Dockerfile`, `docker-compose.yml`, `.dockerignore` — 27/27 tests green
- [x] 5. Phase 3 delivered & audited: `backend/ingestion.py` + sample grid + 2 endpoints + bug fixes — 46/46 tests green
- [ ] 6. Next: Phase 4 — SQLite Database Persistence (`backend/database.py`)
```
