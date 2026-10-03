"""
VATAWARAN Raw Gridded Data Ingestion Engine
============================================
Phase 3 — Bridges raw binary weather data (NetCDF4 / GRIB2 / NCMRWF OpenDAP)
into normalized 9-channel atmospheric state vectors ready for the Spherical
GNN anomaly tracker and Diffusion downscaling pipeline.

State vector (system architecture, TRD §5):
    X_t = [U850, V850, U200, V200, Z500, T850, T2M, Q700, TP]

Climatological standardization (30-year IMDAA / ERA5 baseline,
Indian subcontinent domain 0–45°N, 60–110°E):
    Xhat_t = (X_t - mu_c(d)) / (sigma_c(d) + 1e-6)

Dependencies
------------
Core:     xarray, netCDF4, numpy
Optional: torch (return_torch=True), cfgrib + eccodes (GRIB2 only)
"""

from __future__ import annotations

import math
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:  # xarray is the primary gridded-data reader
    import xarray as xr
except Exception:  # pragma: no cover - env without xarray
    xr = None

try:  # torch conversion is strictly optional (CPU-only wheel in the image)
    import torch
except Exception:  # pragma: no cover - env without torch
    torch = None

# ─── Configuration & Constants ────────────────────────────────────────────────

#: Live NCMRWF NEPS-G 12 km global ensemble stream (OpenDAP / THREDDS).
NCMRWF_OPENDAP_URL = os.getenv(
    "NCMRWF_INGEST_URL",
    "https://opendap.ncmrwf.gov.in/thredds/dodsC/NEPSG/latest.nc",
)

#: Relative (CWD) and vendored-layout (repo root) sample grid locations.
SAMPLE_RELATIVE_PATH = "data/sample/neps_sample.nc"

#: Smallest variance term that keeps standardization numerically stable.
EPSILON = 1e-6

#: Default day-of-year used when the source carries no usable timestamp.
DEFAULT_DOY = 273  # 2026-09-30, matching the prototype BASE_CYCLE

#: Climatological baseline domain (Indian subcontinent & surrounding seas).
DOMAIN: dict[str, Any] = {
    "label": "Indian subcontinent & surrounding seas",
    "lat_bounds": [0.0, 45.0],
    "lon_bounds": [60.0, 110.0],
    "baseline": "30-year IMDAA (NCMRWF) / ERA5 (ECMWF) climatology, 1991-2020",
    "formula": "Xhat = (X - mu_c(d)) / (sigma_c(d) + 1e-6)",
}

#: Canonical channel order of the state vector (MUST match the architecture).
CHANNELS: list[str] = ["U850", "V850", "U200", "V200", "Z500", "T850", "T2M", "Q700", "TP"]

# ─── Channel Catalog & Climatology Baseline ───────────────────────────────────
# mu  : annual-mean climatology over the Indian subcontinent domain
# sigma: annual-mean standard deviation of daily values
# amplitude/peak_doy: harmonic seasonal modulation mu(d) = mu + A*cos(2π(d-day)/365.25)
_EPOCH_DOY = 273  # 30 September — the phase anchor of the annual harmonic

CHANNEL_CATALOG: list[dict[str, Any]] = [
    {
        "key": "U850",
        "name": "Zonal wind at 850 hPa",
        "units": "m/s",
        "level_hpa": 850,
        "description": "Low-level zonal wind (monsoon westerlies over the Arabian Sea & Bay of Bengal)",
        "climatology_mean": 1.5,
        "climatology_std": 4.6,
        "seasonal_amplitude": 2.4,
        "peak_doy": 200,
    },
    {
        "key": "V850",
        "name": "Meridional wind at 850 hPa",
        "units": "m/s",
        "level_hpa": 850,
        "description": "Low-level meridional wind (cross-equatorial monsoon flow)",
        "climatology_mean": 0.2,
        "climatology_std": 2.8,
        "seasonal_amplitude": 1.5,
        "peak_doy": 200,
    },
    {
        "key": "U200",
        "name": "Zonal wind at 200 hPa",
        "units": "m/s",
        "level_hpa": 200,
        "description": "Upper-tropospheric zonal wind (subtropical jet)",
        "climatology_mean": 8.0,
        "climatology_std": 8.5,
        "seasonal_amplitude": 4.0,
        "peak_doy": 15,
    },
    {
        "key": "V200",
        "name": "Meridional wind at 200 hPa",
        "units": "m/s",
        "level_hpa": 200,
        "description": "Upper-tropospheric meridional wind (divergence patterns near the jet)",
        "climatology_mean": 0.0,
        "climatology_std": 6.5,
        "seasonal_amplitude": 3.0,
        "peak_doy": 15,
    },
    {
        "key": "Z500",
        "name": "Geopotential height at 500 hPa",
        "units": "gpm",
        "level_hpa": 500,
        "description": "Mid-tropospheric geopotential height (m^2/s^2 = gpm x 9.80665)",
        "climatology_mean": 5830.0,
        "climatology_std": 95.0,
        "seasonal_amplitude": 55.0,
        "peak_doy": 200,
    },
    {
        "key": "T850",
        "name": "Temperature at 850 hPa",
        "units": "K",
        "level_hpa": 850,
        "description": "Low-tropospheric air temperature",
        "climatology_mean": 285.5,
        "climatology_std": 4.8,
        "seasonal_amplitude": 4.2,
        "peak_doy": 200,
    },
    {
        "key": "T2M",
        "name": "2-metre surface temperature",
        "units": "K",
        "level_hpa": None,
        "description": "Screen-level surface air temperature",
        "climatology_mean": 298.0,
        "climatology_std": 8.2,
        "seasonal_amplitude": 6.5,
        "peak_doy": 140,
    },
    {
        "key": "Q700",
        "name": "Specific humidity at 700 hPa",
        "units": "kg/kg",
        "level_hpa": 700,
        "description": "Mid-level moisture content (kg of water vapour per kg of air)",
        "climatology_mean": 0.0062,
        "climatology_std": 0.0038,
        "seasonal_amplitude": 0.0025,
        "peak_doy": 200,
    },
    {
        "key": "TP",
        "name": "Total precipitation",
        "units": "mm",
        "level_hpa": None,
        "description": "Accumulated total precipitation (1 mm = 1 kg/m^2)",
        "climatology_mean": 7.1,
        "climatology_std": 12.5,
        "seasonal_amplitude": 5.5,
        "peak_doy": 200,
    },
]

CHANNEL_MAP: dict[str, dict[str, Any]] = {c["key"]: c for c in CHANNEL_CATALOG}

#: Name aliases accepted when scanning a raw dataset (case/format-insensitive).
VARIABLE_ALIASES: dict[str, list[str]] = {
    "U850": ["U850", "u850", "U_850", "u_850", "u@850", "u850hpa", "u_850hpa", "var131_850"],
    "V850": ["V850", "v850", "V_850", "v_850", "v@850", "v850hpa", "v_850hpa", "var132_850"],
    "U200": ["U200", "u200", "U_200", "u_200", "u@200", "u200hpa", "u_200hpa", "var131_200"],
    "V200": ["V200", "v200", "V_200", "v_200", "v@200", "v200hpa", "v_200hpa", "var132_200"],
    "Z500": ["Z500", "z500", "Z_500", "z_500", "gh500", "z@500", "gh@500", "z_500hpa", "var129_500", "var156_500"],
    "T850": ["T850", "t850", "T_850", "t_850", "t@850", "temp850", "t_850hpa", "temperature850", "var130_850"],
    "T2M": ["T2M", "T2m", "t2m", "2t", "2mt", "t_2m", "t2metre", "temperature_2m", "temp2m", "var167", "ta2m"],
    "Q700": ["Q700", "q700", "Q_700", "q_700", "q@700", "shum700", "q_700hpa", "hus700", "var133_700"],
    "TP": ["TP", "tp", "TOTPREC", "totprec", "total_precipitation", "tp_tot", "tp_accum", "precip", "var228"],
}

#: Base variables + target level used when a dataset stores level-structured fields
#: (e.g. a variable ``u`` with a ``level`` dimension of size N).
_LEVEL_BASE_MAP: dict[str, tuple[list[str], float]] = {
    "U850": (["u", "u_wind", "uwind", "ucomp", "var131"], 850.0),
    "V850": (["v", "v_wind", "vwind", "vcomp", "var132"], 850.0),
    "U200": (["u", "u_wind", "uwind", "ucomp", "var131"], 200.0),
    "V200": (["v", "v_wind", "vwind", "vcomp", "var132"], 200.0),
    "Z500": (["z", "gh", "geopotential", "hgt", "var129", "var156"], 500.0),
    "T850": (["t", "temp", "temperature", "var130"], 850.0),
    "Q700": (["q", "shum", "hus", "specific_humidity", "humidity", "var133"], 700.0),
}

_MOCK_GRID = {"lat_count": 181, "lon_count": 361, "lat_bounds": (-90.0, 90.0), "lon_bounds": (0.0, 360.0)}


class MissingChannelError(ValueError):
    """Raised when a source grid lacks one or more of the nine required channels."""


# ─── Public Helpers ───────────────────────────────────────────────────────────

def standardize(field: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Apply climatological standardization:  Xhat = (X - mu) / (sigma + 1e-6).

    Non-finite grid cells (and masked cells) are replaced with the climatological mean,
    so a "missing" cell becomes a neutral (zero) anomaly instead of NaN pollution.
    """
    if hasattr(field, "filled"):
        arr = np.asarray(field.filled(np.nan), dtype=np.float64)
    else:
        arr = np.asarray(field, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, mu)
    return (arr - mu) / (sigma + EPSILON)


def _norm_name(name: str) -> str:
    """Lower-cases a variable name and strips non-alphanumeric characters."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _daily_climatology(channel: str, day_of_year: int) -> tuple[float, float]:
    """Returns the day-of-year climatological (mu, sigma) for a channel."""
    base = CHANNEL_MAP[channel]
    phase = 2.0 * math.pi * (day_of_year - base["peak_doy"]) / 365.25
    mu = base["climatology_mean"] + base["seasonal_amplitude"] * math.cos(phase)
    sigma = base["climatology_std"] * (1.0 + 0.06 * math.cos(2.0 * math.pi * (day_of_year - _EPOCH_DOY) / 365.25))
    # Division safety is handled by standardize()'s (sigma + 1e-6); only guard
    # against degenerate sub-float constants here (never clamp physical units).
    return mu, max(sigma, EPSILON)


# ─── Dataset Resolution ───────────────────────────────────────────────────────

def _open_remote_dataset(url: str) -> "xr.Dataset":
    """Streams a remote dataset via OpenDAP; propagates failures for fallback."""
    if xr is None:
        raise ImportError("xarray is required to open remote OpenDAP datasets.")
    old_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(20)
        return xr.open_dataset(url, engine="netcdf4", decode_times=True)
    except Exception as exc:
        raise ValueError(f"Could not connect to remote OpenDAP dataset '{url}': {exc}") from exc
    finally:
        try:
            socket.setdefaulttimeout(old_timeout)
        except Exception:
            pass


def _open_local_dataset(path: Path) -> "xr.Dataset":
    """Opens a local NetCDF4 / (optionally) GRIB2 file."""
    if xr is None:
        raise ImportError("xarray is required to open gridded weather data files.")
    suffix = path.suffix.lower()
    if suffix in (".grib", ".grib2", ".grb", ".grb2"):
        try:
            import cfgrib  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "GRIB2 ingestion requires the optional 'cfgrib' + 'eccodes' stack "
                "(pip install cfgrib, apt-get install libeccodes0 libeccodes-dev)."
            ) from exc
        return xr.open_dataset(path, engine="cfgrib")

    errors: list[str] = []
    try:
        return xr.open_dataset(path, engine="netcdf4", decode_times=True)
    except Exception as exc:
        errors.append(f"netcdf4: {exc}")

    try:
        return xr.open_dataset(path, engine="h5netcdf", decode_times=True)
    except Exception as exc:
        errors.append(f"h5netcdf: {exc}")

    raise ValueError(
        f"Could not open gridded data file '{path.name}': format not recognized or file is corrupted ({'; '.join(errors)})."
    )


def _resolve_sample_path() -> Path:
    """Locates the bundled offline sample grid (env override -> CWD -> repo root)."""
    env = os.getenv("VATAWARAN_SAMPLE_NC")
    if env and Path(env).exists():
        return Path(env)
    rel = Path(SAMPLE_RELATIVE_PATH)
    if rel.exists():
        return rel
    abs_path = Path(__file__).resolve().parent.parent / SAMPLE_RELATIVE_PATH
    if abs_path.exists():
        return abs_path
    raise FileNotFoundError(
        "Bundled offline sample grid not found. Generate it with:\n"
        "    python -c \"from backend.ingestion import create_mock_neps_netcdf; print(create_mock_neps_netcdf())\""
    )


def _resolve_dataset(source: Any) -> tuple[str, str, "xr.Dataset"]:
    """
    Resolves ``source`` into (mode, reference, dataset).

    Modes: ``opendap_live`` (remote stream), ``local_file`` (explicit path),
    ``offline_sample`` (remote failed -> bundled grid fallback).
    """
    if source is None:
        try:
            return "opendap_live", NCMRWF_OPENDAP_URL, _open_remote_dataset(NCMRWF_OPENDAP_URL)
        except Exception:
            sample = _resolve_sample_path()
            return "offline_sample", str(sample), _open_local_dataset(sample)

    text = str(source)
    if text.lower().startswith(("http://", "https://", "dods://")):
        return "opendap_live", text, _open_remote_dataset(text)

    path = Path(text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Gridded data source not found: {path}")
    return "local_file", str(path), _open_local_dataset(path)


# ─── Channel Extraction ───────────────────────────────────────────────────────

def _resolve_level_variable(
    ds: "xr.Dataset", base_names: list[str], level: float
) -> Optional["xr.DataArray"]:
    """Selects the pressure level nearest to ``level`` from a level-structured var."""
    target_set = {_norm_name(b) for b in base_names}
    for vname, da in ds.data_vars.items():
        if _norm_name(vname) not in target_set:
            continue
        for dim in da.dims:
            dim_str = str(dim)
            dim_coords = ds.coords if hasattr(ds, "coords") else {}
            dim_vars = ds.variables if hasattr(ds, "variables") else {}
            if dim not in dim_coords and dim not in dim_vars and dim not in da.coords:
                continue
            dim_lower = dim_str.lower()
            if not any(tok in dim_lower for tok in ("level", "lev", "hpa", "isobaric", "pres", "pressure", "plev")):
                continue
            coord_obj = da[dim] if dim in da.coords else (ds[dim] if dim in ds else None)
            if coord_obj is None:
                continue
            raw_vals = coord_obj.values
            if hasattr(raw_vals, "filled"):
                raw_vals = raw_vals.filled(np.nan)
            values = np.asarray(raw_vals, dtype=float)
            finite_mask = np.isfinite(values)
            if not np.any(finite_mask):
                continue
            unit = str(getattr(coord_obj, "units", "") or "").lower()
            # If units are in Pa (or values exceed 2000 Pa), convert to hPa
            if (any(tok in unit for tok in ("pa", "pascal")) and "hpa" not in unit) or values[finite_mask].max() > 2000.0:
                values = values / 100.0
            diffs = np.abs(values - level)
            min_diff = float(np.nanmin(diffs))
            if min_diff > 100.0:  # Closest level must be within 100 hPa
                continue
            index = int(np.nanargmin(diffs))
            return da.isel({dim: index}).squeeze(drop=True)
    return None


def _find_variable(ds: "xr.Dataset", channel: str) -> tuple[Optional["xr.DataArray"], Optional["xr.DataArray"]]:
    """
    Resolves a canonical channel to a DataArray using aliases or level lookup.
    Returns (resolved_da, raw_source_da_for_unit_metadata).
    """
    # 1) Exact alias matches (fast path for canonical NCMRWF exports)
    for alias in VARIABLE_ALIASES[channel]:
        if alias in ds.data_vars:
            return ds[alias], ds[alias]
    # 2) Format-insensitive alias match (u_850, U850, u850hpa, ...)
    index = {_norm_name(v): v for v in ds.data_vars}
    for alias in VARIABLE_ALIASES[channel]:
        hit = index.get(_norm_name(alias))
        if hit is not None:
            return ds[hit], ds[hit]
    # 3) Level-structured variables (u@850 / z@500 ...)
    if channel in _LEVEL_BASE_MAP:
        base_names, level = _LEVEL_BASE_MAP[channel]
        resolved = _resolve_level_variable(ds, base_names, level)
        if resolved is not None:
            return resolved, resolved
    return None, None


def _conform_channel_units(channel: str, matrix: np.ndarray, da: Optional["xr.DataArray"]) -> np.ndarray:
    """
    Auto-conforms raw meteorological units to the state vector standards:
      * Z500: Geopotential (m^2/s^2) -> Geopotential Height (gpm) via g0 = 9.80665 m/s^2
      * T850 / T2M: Celsius (°C) -> Kelvin (K) (+ 273.15)
      * TP: Accumulated precipitation meters (m) -> millimeters (mm) (* 1000)
    """
    unit = str(getattr(da, "units", "") or "").lower().strip() if da is not None else ""
    finite = matrix[np.isfinite(matrix)]
    mean_val = float(finite.mean()) if finite.size > 0 else 0.0
    max_val = float(finite.max()) if finite.size > 0 else 0.0

    # Z500: Geopotential in m^2/s^2 (~57,000) -> gpm (~5,830)
    if channel == "Z500":
        if any(tok in unit for tok in ("m**2", "m^2", "m2")) or mean_val > 20_000.0:
            return matrix / 9.80665

    # T850 / T2M: Celsius -> Kelvin
    if channel in ("T850", "T2M"):
        if unit in ("c", "degc", "celsius", "degrees_c", "degree_c", "deg_c") or (
            finite.size > 0 and max_val < 100.0 and mean_val < 60.0
        ):
            return matrix + 273.15

    # TP: meters -> mm
    if channel == "TP":
        if unit in ("m", "meter", "metre", "meters", "metres") or (
            finite.size > 0 and max_val < 0.5 and mean_val < 0.05
        ):
            return matrix * 1000.0

    return matrix


def _to_grid_matrix(data_array: "xr.DataArray", channel: str) -> np.ndarray:
    """
    Reduces a DataArray to a 2-D (lat, lon) matrix.
    Singleton/time/ensemble dimensions are dropped; the first member is kept
    for any remaining non-spatial dimension.
    """
    arr = data_array
    # Find spatial dimension names case-insensitively
    lat_dim = next((d for d in arr.dims if str(d).lower() in ("lat", "latitude")), None)
    lon_dim = next((d for d in arr.dims if str(d).lower() in ("lon", "longitude")), None)

    spatial_dims = {d for d in (lat_dim, lon_dim) if d is not None}
    if len(spatial_dims) < 2 and arr.ndim > 2:
        spatial_dims = set(list(arr.dims)[-2:])
        if lat_dim is None:
            lat_dim = list(arr.dims)[-2]
        if lon_dim is None:
            lon_dim = list(arr.dims)[-1]

    # Drop singletons first
    for dim in list(arr.dims):
        if dim not in spatial_dims and arr.sizes[dim] == 1:
            arr = arr.squeeze(dim=dim, drop=True)

    # Slice first member of any remaining non-spatial dimension (time, ensemble, etc.)
    for dim in list(arr.dims):
        if dim not in spatial_dims:
            arr = arr.isel({dim: 0})

    # Reorder to (lat, lon)
    if lat_dim and lon_dim and lat_dim in arr.dims and lon_dim in arr.dims:
        arr = arr.transpose(lat_dim, lon_dim)

    raw_vals = arr.values
    if hasattr(raw_vals, "filled"):
        raw_vals = raw_vals.filled(np.nan)
    matrix = np.asarray(raw_vals, dtype=np.float64)
    if matrix.ndim != 2:
        raise MissingChannelError(
            f"Channel {channel}: expected a 2-D lat/lon grid, got shape {matrix.shape}"
        )
    return matrix


def _extract_channels(ds: "xr.Dataset") -> tuple[dict[str, np.ndarray], list[str]]:
    """Extracts all nine channels; returns (raw_fields, missing_channels)."""
    raw: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for channel in CHANNELS:
        found, meta_da = _find_variable(ds, channel)
        if found is None:
            missing.append(channel)
            continue
        matrix = _to_grid_matrix(found, channel)
        matrix = _conform_channel_units(channel, matrix, meta_da)
        raw[channel] = matrix
    return raw, missing


# ─── Metadata Helpers ─────────────────────────────────────────────────────────

def _coord_meta(ds: "xr.Dataset", name: str):
    lat_candidates = ("lat", "latitude", "LAT", "LATITUDE", "Latitude", "Lat")
    lon_candidates = ("lon", "longitude", "LON", "LONGITUDE", "Longitude", "Lon")
    candidates = lat_candidates if name == "lat" else lon_candidates if name == "lon" else (name,)
    for candidate in candidates:
        if candidate not in ds.coords and candidate not in ds.variables:
            continue
        raw_val = ds[candidate].values
        if hasattr(raw_val, "filled"):
            raw_val = raw_val.filled(np.nan)
        values = np.asarray(raw_val, dtype=float).reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        return {
            "units": str(getattr(ds[candidate], "units", "") or ""),
            "min": round(float(finite.min()), 6),
            "max": round(float(finite.max()), 6),
            "step": round(float(np.diff(values).mean()), 6) if values.size > 1 else 0.0,
            "count": int(values.size),
        }
    return None


def _grid_meta(ds: "xr.Dataset") -> dict[str, Any]:
    """Extracts grid bounds / step / count for the standardized tensor axes."""
    return {"lat": _coord_meta(ds, "lat"), "lon": _coord_meta(ds, "lon")}


def _extract_timestamps(ds: "xr.Dataset") -> list[str]:
    """Returns ISO-8601 UTC timestamps decoded from the source (up to 3)."""
    if "time" not in ds.coords and "time" not in ds.variables:
        return []
    time_var = ds["time"]
    values = np.atleast_1d(np.asarray(time_var.values))
    stamps: list[str] = []
    for value in values.ravel()[:3]:
        try:
            if hasattr(value, "strftime"):  # cftime / python datetime objects
                stamps.append(value.strftime("%Y-%m-%dT%H:%M:%SZ"))
                continue
            if isinstance(value, np.datetime64):
                dt = value.astype("datetime64[s]").item()
                if dt is not None and hasattr(dt, "strftime"):
                    stamps.append(dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
                    continue
            numer = int(value)
            if numer > 10_000_000_000_000:  # nanoseconds since 1970-01-01
                dt = datetime.fromtimestamp(numer / 1e9, tz=timezone.utc)
            else:  # seconds since 1970-01-01
                dt = datetime.fromtimestamp(numer, tz=timezone.utc)
            stamps.append(dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        except Exception:
            stamps.append(str(value))
    return stamps


def _day_of_year(iso_stamp: str) -> int:
    """Day-of-year from the first timestamp; falls back to the cycle anchor."""
    try:
        normalized = iso_stamp.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timetuple().tm_yday
    except Exception:
        return DEFAULT_DOY


def _stats(values: np.ndarray) -> dict[str, float]:
    if hasattr(values, "filled"):
        finite = np.asarray(values.filled(np.nan), dtype=np.float64)
    else:
        finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": round(float(finite.min()), 6),
        "max": round(float(finite.max()), 6),
        "mean": round(float(finite.mean()), 6),
        "std": round(float(finite.std()), 6),
    }


# ─── Primary API ──────────────────────────────────────────────────────────────

def ingest_ncmrwf_cycle(
    source: Optional[str | Path] = None,
    return_torch: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Ingests an NCMRWF model cycle from a local NetCDF4/GRIB2 file or a live
    OpenDAP stream and normalizes it into a 9-channel atmospheric state vector.

    Parameters
    ----------
    source:
        Local file path, remote OpenDAP URL, or ``None``. When ``None`` the
        engine first tries the live NCMRWF stream and, on any connection
        failure, gracefully falls back to the bundled offline sample grid.
    return_torch:
        When True, returns ``torch.Tensor`` instead of ``numpy.ndarray``.

    Returns
    -------
    state_vector:
        Normalized numpy (or torch) array of shape ``(9, H, W)`` in the
        canonical order ``[U850, V850, U200, V200, Z500, T850, T2M, Q700, TP]``.
    metadata:
        Dictionary with grid bounds, timestamps, channel names, climatological
        baseline statistics, per-channel distribution stats and warnings.
    """
    if xr is None:
        raise ImportError("xarray is required by backend.ingestion — pip install xarray")

    mode, source_ref, ds = _resolve_dataset(source)

    try:
        raw, missing = _extract_channels(ds)
        if missing:
            raise MissingChannelError(
                f"Dataset is missing required channels: {', '.join(missing)}. "
                f"Available variables: {sorted(ds.data_vars)}. "
                "Refusing to fabricate state-vector channels."
            )

        timestamps = _extract_timestamps(ds)
        day_of_year = _day_of_year(timestamps[0]) if timestamps else DEFAULT_DOY
        source_attrs = {
            "title": str(ds.attrs.get("title", "") or ""),
            "model": str(ds.attrs.get("model", "") or ""),
        }

        shapes = {v.shape for v in raw.values()}
        if len(shapes) != 1:
            raise MissingChannelError(f"Channel grids must share one spatial shape; got {sorted(shapes)}")
        height, width = next(iter(shapes))

        state = np.empty((len(CHANNELS), height, width), dtype=np.float32)
        baseline: dict[str, dict[str, Any]] = {}
        raw_stats: dict[str, dict[str, float]] = {}
        norm_stats: dict[str, dict[str, float]] = {}
        imputed_cells: dict[str, int] = {}

        for index, channel in enumerate(CHANNELS):
            mu, sigma = _daily_climatology(channel, day_of_year)
            baseline[channel] = {
                "climatology_mean": round(mu, 6),
                "climatology_std": round(sigma, 6),
                "units": CHANNEL_MAP[channel]["units"],
                "level_hpa": CHANNEL_MAP[channel]["level_hpa"],
            }
            field = raw[channel]
            imputed_cells[channel] = int(np.count_nonzero(~np.isfinite(field)))
            standardized = standardize(field, mu, sigma)
            state[index] = standardized.astype(np.float32)
            raw_stats[channel] = _stats(field)
            norm_stats[channel] = _stats(standardized)

        metadata: dict[str, Any] = {
            "source": source_ref,
            "source_mode": mode,
            "channels": list(CHANNELS),
            "state_vector_shape": [len(CHANNELS), height, width],
            "dtype": "float32",
            "grid": _grid_meta(ds),
            "timestamps": timestamps,
            "day_of_year": day_of_year,
            "source_attrs": source_attrs,
            "domain": DOMAIN,
            "standardization": "Xhat = (X - mu_c(d)) / (sigma_c(d) + 1e-6)",
            "baseline": baseline,
            "raw_stats": raw_stats,
            "normalized_stats": norm_stats,
            "imputed_cells": imputed_cells,
            "warnings": (
                []
                if mode == "offline_sample"
                else [f"Non-finite grid cells imputed per channel: {imputed_cells}"]
                if any(imputed_cells.values())
                else []
            ),
        }
    finally:
        try:
            ds.close()
        except Exception:
            pass

    if return_torch:
        if torch is None:
            raise ImportError(
                "return_torch=True requires PyTorch, which is not installed in this environment."
            )
        return torch.from_numpy(state), metadata
    return state, metadata


# ─── Offline Sample Generator ─────────────────────────────────────────────────

def _synthetic_field(channel: str, rng: np.random.Generator) -> np.ndarray:
    """
    Generates a physically plausible global 1-degree field for one channel.

    Fields follow textbook structure of each variable over the globe so the
    sample grid exercises realistic gradients while staying lightweight.
    """
    lat = np.linspace(*_MOCK_GRID["lat_bounds"], _MOCK_GRID["lat_count"]).reshape(-1, 1)
    lon = np.linspace(*_MOCK_GRID["lon_bounds"], _MOCK_GRID["lon_count"]).reshape(1, -1)
    phi = np.radians(lat)
    lam = np.radians(lon)
    cos_phi = np.cos(phi)
    shape = (181, 361)

    if channel == "U850":
        return (6.5 * np.sin(2 * phi) + 2.2 * np.sin(lam) * cos_phi + rng.normal(0, 0.55, shape)).astype("f4")
    if channel == "V850":
        return (2.6 * np.cos(2 * phi) - 1.4 * np.sin(lam) * cos_phi + rng.normal(0, 0.45, shape)).astype("f4")
    if channel == "U200":
        return (6.0 + 27.0 * np.sin(2 * phi) ** 2 + 3.0 * np.sin(lam - 1.5) * cos_phi**2 + rng.normal(0, 0.9, shape)).astype("f4")
    if channel == "V200":
        return (5.5 * np.cos(2 * phi) + 2.0 * np.cos(lam) * cos_phi + rng.normal(0, 0.8, shape)).astype("f4")
    if channel == "Z500":
        return (4950.0 + 900.0 * cos_phi + 45.0 * np.sin(4 * lam) * cos_phi + rng.normal(0, 12.0, shape)).astype("f4")
    if channel == "T850":
        return (258.0 + 30.0 * cos_phi + 2.2 * np.sin(lam) + rng.normal(0, 0.6, shape)).astype("f4")
    if channel == "T2M":
        return (250.0 + 52.0 * cos_phi + 2.6 * np.sin(lam + 1.0) + rng.normal(0, 0.9, shape)).astype("f4")
    if channel == "Q700":
        # Monsoon-axis moisture gradient (higher near the Bay of Bengal limb)
        monsoon_axis = 0.0016 * np.sin(lam - 1.3) * cos_phi
        return np.maximum(0.0003, 0.0085 * cos_phi + monsoon_axis + rng.normal(0, 0.0004, shape)).astype("f4")
    if channel == "TP":
        monsoon_band = np.clip(np.sin(np.radians(lat - 5.0) * 2.2), 0.0, 1.0)
        bay_of_bengal_boost = 0.65 + 0.35 * np.cos(np.radians(lon - 75.0))
        base = (1.2 + 13.0 * cos_phi * monsoon_band) * bay_of_bengal_boost
        multiplicative_noise = np.exp(rng.normal(0.0, 0.45, shape))
        return np.maximum(0.0, base * multiplicative_noise).astype("f4")
    raise ValueError(f"Unknown channel: {channel}")


def create_mock_neps_netcdf(output_path: str = SAMPLE_RELATIVE_PATH) -> str:
    """
    Generates a realistic lightweight NEPS-G sample NetCDF4 grid with all nine
    atmospheric channels plus coordinate dimensions (lat, lon, time, level) for
    offline tests, development and Docker smoke runs.
    """
    try:
        import netCDF4 as nc4
    except ImportError as exc:
        raise ImportError("netCDF4 python bindings are required to generate the sample grid.") from exc

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        rng = np.random.default_rng(26078)  # deterministic: PS ID as seed
        with nc4.Dataset(str(path), "w", format="NETCDF4") as ds:
            ds.title = "VATAWARAN NEPS-G 12 km sample grid (offline verification)"
            ds.model = "NEPS-G 12km (mock)"
            ds.institution = "MoES / NCMRWF (simulated)"
            ds.forecast_reference_time = "2026-09-30T00:00:00Z"
            ds.grid_resolution_km = 12.0

            # Coordinate dimensions
            ds.createDimension("time", 1)
            ds.createDimension("lat", _MOCK_GRID["lat_count"])
            ds.createDimension("lon", _MOCK_GRID["lon_count"])
            ds.createDimension("level", 4)

            time_var = ds.createVariable("time", "i8", ("time",))
            time_var.units = "hours since 2026-09-30 00:00:00"
            time_var.calendar = "standard"
            time_var.long_name = "forecast_reference_time"
            time_var[:] = 0

            lat_var = ds.createVariable("lat", "f4", ("lat",))
            lat_var.units = "degrees_north"
            lat_var.long_name = "latitude"
            lat_var[:] = np.linspace(*_MOCK_GRID["lat_bounds"], _MOCK_GRID["lat_count"])

            lon_var = ds.createVariable("lon", "f4", ("lon",))
            lon_var.units = "degrees_east"
            lon_var.long_name = "longitude"
            lon_var[:] = np.linspace(*_MOCK_GRID["lon_bounds"], _MOCK_GRID["lon_count"])

            level_var = ds.createVariable("level", "i4", ("level",))
            level_var.units = "hPa"
            level_var.positive = "down"
            level_var[:] = [850, 700, 500, 200]

            # Nine atmospheric channels (zlib-compressed to keep the file tiny)
            for channel in CHANNELS:
                info = CHANNEL_MAP[channel]
                var = ds.createVariable(channel, "f4", ("time", "lat", "lon"), zlib=True, complevel=4, fill_value=None)
                var.units = info["units"]
                var.long_name = info["name"]
                var.standard_name = info["name"]
                var.level_hpa = info["level_hpa"] if info["level_hpa"] else "surface"
                var[:] = _synthetic_field(channel, rng)[np.newaxis, :, :]

    return str(path)