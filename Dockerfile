# ================================================================================
# PROJECT VATAWARAN — Production Container Image
# AI-Driven Spatio-Temporal Extreme Weather Tracking & Downscaling Engine
# Smart India Hackathon 2026 | Problem Statement ID: 26078
# ================================================================================
#
# Build :   docker build -t vatavaran-app .
# Run   :   docker run -p 8000:8000 --env-file .env vatavaran-app
# Compose:  docker compose up --build -d
#
# Why a SINGLE stage? Every Python dependency in requirements.txt (torch,
# cartopy, netCDF4, xarray, metpy, ...) ships prebuilt manylinux wheels, so
# there are no C sources to compile in a separate builder stage. A single,
# unprivileged stage keeps the image smaller and easier to debug than a
# classical multi-stage split with no compiler work to separate.
#
# Security posture:
#   * Runs as an unprivileged system user (never root).
#   * .dockerignore keeps secrets (.env), VCS history, and venvs out of the
#     build context.
#   * apt + pip caches are purged inside the layer that creates them.

FROM python:3.11-slim-bookworm

# --------------------------------------------------------------------------------
# Image metadata
# --------------------------------------------------------------------------------
LABEL org.opencontainers.image.title="VATAWARAN API"
LABEL org.opencontainers.image.description="AI-Driven Spatio-Temporal Extreme Weather Tracking & Downscaling Engine (SIH 2026 | PS 26078)"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/dr4979555-bot/Project_VATAVARAN.git"

# --------------------------------------------------------------------------------
# Runtime behaviour
# --------------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# --------------------------------------------------------------------------------
# System dependencies
# --------------------------------------------------------------------------------
#  * curl          — used by the container healthcheck (and ad-hoc debugging)
#  * ca-certificates — TLS trust store for HTTPS feeds (NCMRWF, IMD, MOSDAC, NDMA)
#  * gcc / build-essential — native toolchain, in case any sdist needs compiling
#  * Geospatial C libraries:
#      libgdal-dev      — GDAL (raster/vector geodata)
#      libgeos-dev      — GEOS geometry engine (required by cartopy at runtime)
#      libproj-dev      — PROJ cartographic projections
#      proj-data        — PROJ datum/coordinate metadata
#      libnetcdf-dev    — netCDF C library (netCDF4 Python bindings)
#      libhdf5-dev      — HDF5 (h5py, h5netcdf, netCDF4 I/O)
#      libudunits2-0    — UDUNITS unit conversions (MetPy/cf-units)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gcc \
        build-essential \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        proj-data \
        libnetcdf-dev \
        libhdf5-dev \
        libudunits2-0 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------------
# Python dependencies (cache layers: requirements.txt changes less often than code)
# --------------------------------------------------------------------------------
COPY requirements.txt ./

# Install CPU-only PyTorch FIRST so that `pip install -r requirements.txt` never
# resolves the ~2.5 GB CUDA wheel from PyPI. The application stays fully
# functional without GPU wheels — backend/main.py loads every optional ML module
# through _load_optional_attr() and degrades gracefully when it is missing.
# (Verified 2026-10: download.pytorch.org/whl/cpu serves torch's dependencies too.)
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.3.0" \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# --------------------------------------------------------------------------------
# Application code
# --------------------------------------------------------------------------------
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY tests/ ./tests/
COPY data/sample/ ./data/sample/
COPY README.md ROADMAP.md ./

# --------------------------------------------------------------------------------
# Least-privilege runtime user
# --------------------------------------------------------------------------------
# Unprivileged system user that owns the application tree. The app is read-only
# in Phase 2; Phase 4 (SQLite persistence) will layer a writable named volume on
# top of /app without ever needing root.
RUN useradd --system --no-create-home --uid 8000 vatavaran \
    && chown -R vatavaran:vatavaran /app

USER vatavaran

EXPOSE 8000

# Dockerfile-level healthcheck (docker-compose.yml mirrors this definition).
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/v1/system/status || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]