"""
Phase 3 automated tests for backend/ingestion.py.
Verifies the 9-channel ingestion contract:
  * Offline NetCDF4 sample grid -> state vector (9, 181, 361)
  * No NaN / Inf in standardized outputs, correct channel order/units
  * Metadata (grid bounds, timestamps, IMDAA/ERA5 baseline)
  * Graceful fallback when the remote NCMRWF OpenDAP stream is unreachable
  * PyTorch tensor conversion, and the Phase 3 REST endpoints
"""

from fastapi.testclient import TestClient

import numpy as np
import pytest

from backend import ingestion as ing
from backend.main import app

SAMPLE = "data/sample/neps_sample.nc"
client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_grid():
    """Guarantees the offline sample grid exists before any ingestion test."""
    import os

    if not os.path.exists(SAMPLE):
        ing.create_mock_neps_netcdf(SAMPLE)
    return SAMPLE


# ── Sample grid generator ─────────────────────────────────────────────────────


def test_create_mock_neps_netcdf_structure(ensure_sample_grid):
    """Verify the generated NetCDF carries coords + all nine channels, and is small."""
    import os

    import xarray as xr

    out = ing.create_mock_neps_netcdf(SAMPLE)  # idempotent regeneration
    assert os.path.exists(out)
    assert os.path.getsize(out) < 5 * 1024 * 1024  # "excluded from bloat" bound

    ds = xr.open_dataset(out)
    try:
        assert ds.sizes["time"] == 1
        assert ds.sizes["lat"] == 181
        assert ds.sizes["lon"] == 361
        assert ds.sizes["level"] == 4
        assert list(ds.data_vars) == ing.CHANNELS
        assert "level" in ds
    finally:
        ds.close()


# ── Core ingestion contract ───────────────────────────────────────────────────


def test_ingest_offline_sample_shape_and_finite():
    """Verify tensor output shape is exactly (9, H, W) with no NaN/Inf values."""
    state, meta = ing.ingest_ncmrwf_cycle(SAMPLE)
    assert isinstance(state, np.ndarray)
    assert state.shape == (9, 181, 361)
    assert state.dtype == np.float32
    assert np.isfinite(state).all(), "standardized state vector must be finite"
    assert meta["state_vector_shape"] == [9, 181, 361]


def test_ingest_channel_order_and_units():
    """Verify the 9 channels appear in the exact architecture order with spec units."""
    catalog = ing.CHANNEL_CATALOG
    assert [c["key"] for c in catalog] == ["U850", "V850", "U200", "V200", "Z500", "T850", "T2M", "Q700", "TP"]
    units = {c["key"]: c["units"] for c in catalog}
    assert units["U850"] == units["V850"] == units["U200"] == units["V200"] == "m/s"
    assert units["Z500"] in ("gpm", "m^2/s^2")
    assert units["T850"] == units["T2M"] == "K"
    assert units["Q700"] == "kg/kg"
    assert units["TP"] in ("mm", "kg/m^2")

    state, meta = ing.ingest_ncmrwf_cycle(SAMPLE)
    assert meta["channels"] == ing.CHANNELS
    assert list(meta["baseline"].keys()) == ing.CHANNELS
    for channel in ing.CHANNELS:
        assert "climatology_mean" in meta["baseline"][channel]
        assert "climatology_std" in meta["baseline"][channel]
        assert meta["baseline"][channel]["units"] == units[channel]


def test_ingest_metadata_grid_bounds_and_timestamps():
    """Verify grid bounds, timestamps, and source metadata are reported."""
    _, meta = ing.ingest_ncmrwf_cycle(SAMPLE)
    assert meta["source_mode"] == "local_file"
    lat = meta["grid"]["lat"]
    lon = meta["grid"]["lon"]
    assert lat["min"] == -90.0 and lat["max"] == 90.0 and lat["count"] == 181
    assert lon["min"] == 0.0 and lon["max"] == 360.0 and lon["count"] == 361
    assert len(meta["timestamps"]) >= 1
    assert meta["timestamps"][0].startswith("2026-09-30")
    assert meta["day_of_year"] == 273
    assert meta["domain"]["lat_bounds"] == [0.0, 45.0]
    assert meta["domain"]["lon_bounds"] == [60.0, 110.0]
    assert "Xhat" in meta["standardization"]


def test_ingest_standardized_values_plausible():
    """Sanity-check standardized anomalies: finite, small means, sane spread."""
    state, meta = ing.ingest_ncmrwf_cycle(SAMPLE)

    for channel in ing.CHANNELS:
        idx = ing.CHANNELS.index(channel)
        norm = state[idx]
        assert np.abs(norm.mean()) < 8.0, f"{channel} normalized mean out of range"
        assert 0.01 <= norm.std() <= 15.0, f"{channel} normalized std out of range"

    # Indian-subdomain slice (0-45N, 60-110E): anomalies must stay moderate.
    # Grid layout is lat=-90..90 (row 90 == 0N, row 135 == 45N), lon=0..360.
    india = state[:, 90:136, 60:111]
    assert india.shape[1:] == (46, 51)
    assert np.isfinite(india).all()
    for channel in ing.CHANNELS:
        anomaly = india[ing.CHANNELS.index(channel)]
        assert np.abs(anomaly.mean()) < 6.0, f"{channel}: Indian-domain mean anomaly out of range"
        assert anomaly.std() > 0.05, f"{channel}: Indian-domain field looks degenerate"


def test_ingest_remote_failure_falls_back_to_sample(monkeypatch):
    """OpenDAP outage must degrade gracefully to the bundled offline grid."""
    def boom(url):
        raise ConnectionError("simulated NCMRWF OpenDAP outage")

    monkeypatch.setattr(ing, "_open_remote_dataset", boom)
    state, meta = ing.ingest_ncmrwf_cycle(source=None)
    assert meta["source_mode"] == "offline_sample"
    assert state.shape == (9, 181, 361)
    assert np.isfinite(state).all()


def test_ingest_missing_file_raises():
    """Explicit but non-existent paths must fail loudly, never silently fabricate."""
    with pytest.raises(FileNotFoundError):
        ing.ingest_ncmrwf_cycle("data/sample/definitely_missing.nc")


def test_ingest_return_torch_tensor():
    """Verify optional PyTorch conversion keeps shape/dtype contracts."""
    state, meta = ing.ingest_ncmrwf_cycle(SAMPLE, return_torch=True)
    import torch

    assert isinstance(state, torch.Tensor)
    assert state.shape == (9, 181, 361)
    assert state.dtype == torch.float32
    assert bool(torch.isfinite(state).all())


def test_ingest_missing_channel_raises(tmp_path):
    """A grid missing any of the nine channels must be rejected."""
    import netCDF4 as nc4

    import warnings

    partial = tmp_path / "partial.nc"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with nc4.Dataset(str(partial), "w", format="NETCDF4") as ds:
            ds.createDimension("time", 1)
            ds.createDimension("lat", 10)
            ds.createDimension("lon", 10)
            lat = ds.createVariable("lat", "f4", ("lat",))
            lat[:] = np.arange(10)
            lon = ds.createVariable("lon", "f4", ("lon",))
            lon[:] = np.arange(10)
            t2m = ds.createVariable("T2M", "f4", ("time", "lat", "lon"))
            t2m[:] = np.full((1, 10, 10), 300.0, dtype=np.float32)

    with pytest.raises(ing.MissingChannelError) as excinfo:
        ing.ingest_ncmrwf_cycle(str(partial))
    message = str(excinfo.value)
    assert "U850" in message and "TP" in message
    assert issubclass(ing.MissingChannelError, ValueError)


# ── REST API endpoints (Phase 3) ──────────────────────────────────────────────


def test_ingest_api_channels_endpoint():
    """GET /api/v1/ingest/channels exposes names, units and baseline statistics."""
    response = client.get("/api/v1/ingest/channels")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["count"] == 9
    assert data["order"] == ing.CHANNELS
    assert len(data["channels"]) == 9
    assert data["channels"][0]["key"] == "U850"
    assert all("units" in c and "climatology_mean" in c for c in data["channels"])


def test_ingest_api_cycle_json():
    """POST /api/v1/ingest/cycle with a path-based JSON payload."""
    response = client.post("/api/v1/ingest/cycle", json={"source": SAMPLE})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ingested"
    assert data["state_vector_shape"] == [9, 181, 361]
    assert data["source_mode"] == "local_file"
    assert "baseline" in data and "grid" in data and "timestamps" in data


def test_ingest_api_cycle_upload():
    """POST /api/v1/ingest/cycle accepts a multipart NetCDF upload."""
    with open(SAMPLE, "rb") as handle:
        response = client.post(
            "/api/v1/ingest/cycle",
            files={"file": ("neps_sample.nc", handle, "application/octet-stream")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ingested"
    assert data["source_mode"] == "local_file"
    assert data["state_vector_shape"] == [9, 181, 361]


def test_ingest_api_cycle_missing_source():
    """A missing source path must surface as a 4xx client error, not a crash."""
    response = client.post("/api/v1/ingest/cycle", json={"source": "data/sample/nope.nc"})
    assert response.status_code == 400


def test_ingest_api_cycle_corrupted_file_upload():
    """Uploading a corrupted non-NetCDF file must return 400 Bad Request, not 500."""
    response = client.post(
        "/api/v1/ingest/cycle",
        files={"file": ("corrupt.nc", b"not_a_valid_netcdf_header", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "not a valid NetCDF4" in response.json()["detail"] or "format not recognized" in response.json()["detail"]


def test_ingest_api_cycle_empty_file_upload():
    """Uploading an empty file must return 400 Bad Request with a clear message."""
    response = client.post(
        "/api/v1/ingest/cycle",
        files={"file": ("empty.nc", b"", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_ingest_case_insensitive_spatial_dimensions(tmp_path):
    """Ensure coordinates and dimensions named 'Latitude'/'Longitude' resolve cleanly."""
    import netCDF4 as nc4
    import warnings

    test_file = tmp_path / "case_test.nc"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with nc4.Dataset(str(test_file), "w", format="NETCDF4") as ds:
            ds.createDimension("time", 1)
            ds.createDimension("Latitude", 10)
            ds.createDimension("Longitude", 10)
            lat = ds.createVariable("Latitude", "f4", ("Latitude",))
            lat[:] = np.linspace(10, 20, 10)
            lon = ds.createVariable("Longitude", "f4", ("Longitude",))
            lon[:] = np.linspace(70, 80, 10)

            for channel in ing.CHANNELS:
                v = ds.createVariable(channel, "f4", ("time", "Latitude", "Longitude"))
                v[:] = np.full((1, 10, 10), 5.0, dtype=np.float32)

    state, meta = ing.ingest_ncmrwf_cycle(str(test_file))
    assert state.shape == (9, 10, 10)
    assert meta["grid"]["lat"] is not None
    assert meta["grid"]["lon"] is not None
    assert meta["grid"]["lat"]["count"] == 10
    assert meta["grid"]["lon"]["count"] == 10


def test_ingest_physical_unit_autoconformance(tmp_path):
    """Verify Z500 m^2/s^2 -> gpm, T2M °C -> K, and TP m -> mm auto-conformance."""
    import netCDF4 as nc4
    import warnings

    test_file = tmp_path / "units_test.nc"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with nc4.Dataset(str(test_file), "w", format="NETCDF4") as ds:
            ds.createDimension("time", 1)
            ds.createDimension("lat", 5)
            ds.createDimension("lon", 5)
            lat = ds.createVariable("lat", "f4", ("lat",))
            lat[:] = np.linspace(15, 20, 5)
            lon = ds.createVariable("lon", "f4", ("lon",))
            lon[:] = np.linspace(75, 80, 5)

            for channel in ing.CHANNELS:
                v = ds.createVariable(channel, "f4", ("time", "lat", "lon"))
                if channel == "Z500":
                    v.units = "m^2/s^2"
                    v[:] = np.full((1, 5, 5), 5830.0 * 9.80665, dtype=np.float32)
                elif channel in ("T850", "T2M"):
                    v.units = "degC"
                    v[:] = np.full((1, 5, 5), 25.0, dtype=np.float32)
                elif channel == "TP":
                    v.units = "m"
                    v[:] = np.full((1, 5, 5), 0.010, dtype=np.float32)  # 10 mm
                else:
                    v[:] = np.full((1, 5, 5), 1.0, dtype=np.float32)

    state, meta = ing.ingest_ncmrwf_cycle(str(test_file))
    # Raw stats should reflect converted physical units:
    assert 5800.0 < meta["raw_stats"]["Z500"]["mean"] < 5850.0  # Converted to gpm
    assert 298.0 < meta["raw_stats"]["T2M"]["mean"] < 299.0     # 25 C -> 298.15 K
    assert 9.9 < meta["raw_stats"]["TP"]["mean"] < 10.1         # 0.01 m -> 10 mm
    # Standardized values must stay moderate:
    assert np.abs(state[ing.CHANNELS.index("Z500")]).mean() < 1.0


def test_ingest_masked_array_missing_values(tmp_path):
    """Masked array fill values must become neutral zero anomalies and not explode."""
    import netCDF4 as nc4
    import warnings

    test_file = tmp_path / "masked_test.nc"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with nc4.Dataset(str(test_file), "w", format="NETCDF4") as ds:
            ds.createDimension("time", 1)
            ds.createDimension("lat", 4)
            ds.createDimension("lon", 4)
            lat = ds.createVariable("lat", "f4", ("lat",))
            lat[:] = np.linspace(10, 15, 4)
            lon = ds.createVariable("lon", "f4", ("lon",))
            lon[:] = np.linspace(70, 75, 4)

            for channel in ing.CHANNELS:
                v = ds.createVariable(channel, "f4", ("time", "lat", "lon"), fill_value=9.96921e36)
                data = np.full((1, 4, 4), 10.0, dtype=np.float32)
                data[0, 0, 0] = 9.96921e36  # Missing cell
                v[:] = data

    state, meta = ing.ingest_ncmrwf_cycle(str(test_file))
    assert np.isfinite(state).all()
    # Masked cell at [0, 0] was replaced by climatological mean -> standardized anomaly == 0.0
    for idx in range(len(ing.CHANNELS)):
        assert np.isclose(state[idx, 0, 0], 0.0, atol=1e-4)
    assert meta["imputed_cells"]["T2M"] == 1


def test_ingest_ecmwf_grib_aliases(tmp_path):
    """Verify var131_850 (U) and var132_850 (V) ECMWF aliases are recognized."""
    import netCDF4 as nc4
    import warnings

    test_file = tmp_path / "grib_alias_test.nc"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with nc4.Dataset(str(test_file), "w", format="NETCDF4") as ds:
            ds.createDimension("time", 1)
            ds.createDimension("lat", 3)
            ds.createDimension("lon", 3)
            lat = ds.createVariable("lat", "f4", ("lat",))
            lat[:] = [10, 11, 12]
            lon = ds.createVariable("lon", "f4", ("lon",))
            lon[:] = [70, 71, 72]

            alias_map = {
                "var131_850": 3.5,  # U850
                "var132_850": 1.2,  # V850
                "U200": 8.0,
                "V200": 0.0,
                "var129_500": 5830.0,  # Z500
                "T850": 285.0,
                "T2M": 298.0,
                "Q700": 0.006,
                "TP": 5.0,
            }
            for name, val in alias_map.items():
                v = ds.createVariable(name, "f4", ("time", "lat", "lon"))
                v[:] = np.full((1, 3, 3), val, dtype=np.float32)

    state, meta = ing.ingest_ncmrwf_cycle(str(test_file))
    assert state.shape == (9, 3, 3)
    assert np.isclose(meta["raw_stats"]["U850"]["mean"], 3.5, atol=1e-2)
    assert np.isclose(meta["raw_stats"]["V850"]["mean"], 1.2, atol=1e-2)


def test_ingest_declared_units_beat_value_heuristic(tmp_path):
    """
    Declared canonical units must NEVER be overridden by the value-based
    heuristics inside `_conform_channel_units`.

    Regression: a TP field legitimately declared in millimetres but with light
    rainfall (mean < 0.05) used to be silently multiplied by 1000 because its
    values "looked like" metres, corrupting the state vector into a false
    precipitation anomaly.
    """
    import warnings

    import netCDF4 as nc4

    test_file = tmp_path / "declared_units_win.nc"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with nc4.Dataset(str(test_file), "w", format="NETCDF4") as ds:
            ds.createDimension("time", 1)
            ds.createDimension("lat", 3)
            ds.createDimension("lon", 3)
            lat = ds.createVariable("lat", "f4", ("lat",))
            lat[:] = [10.0, 11.0, 12.0]
            lon = ds.createVariable("lon", "f4", ("lon",))
            lon[:] = [70.0, 71.0, 72.0]

            for channel in ing.CHANNELS:
                v = ds.createVariable(channel, "f4", ("time", "lat", "lon"))
                if channel == "TP":
                    v.units = "mm"
                    v[:] = np.full((1, 3, 3), 0.02, dtype="f4")  # light but canonical mm
                elif channel in ("T850", "T2M"):
                    v.units = "K"
                    v[:] = np.full((1, 3, 3), 285.0, dtype="f4")
                elif channel == "Z500":
                    v.units = "gpm"
                    v[:] = np.full((1, 3, 3), 5830.0, dtype="f4")
                else:
                    v[:] = np.full((1, 3, 3), 1.0, dtype="f4")

    _, meta = ing.ingest_ncmrwf_cycle(str(test_file))
    assert meta["raw_stats"]["TP"]["mean"] == pytest.approx(0.02, abs=1e-4)  # NOT 20.0
    assert 284 <= meta["raw_stats"]["T2M"]["mean"] <= 286  # NOT 558.15 (285 K kept as K)
    assert 5825 <= meta["raw_stats"]["Z500"]["mean"] <= 5835  # NOT ~594 (5830 gpm kept)


def test_ingest_masked_coordinate_metadata(tmp_path):
    """Masked coordinate cells must not leak fill values into grid metadata."""
    import warnings

    import netCDF4 as nc4

    test_file = tmp_path / "masked_coord.nc"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with nc4.Dataset(str(test_file), "w", format="NETCDF4") as ds:
            ds.createDimension("time", 1)
            ds.createDimension("lat", 4)
            ds.createDimension("lon", 4)
            lat = ds.createVariable("lat", "f4", ("lat",), fill_value=-9999.0)
            lat.units = "degrees_north"
            lat[:] = np.ma.array([10.0, 15.0, 20.0, 25.0], mask=[False, False, True, False])
            lon = ds.createVariable("lon", "f4", ("lon",))
            lon[:] = np.linspace(70.0, 73.0, 4)
            for channel in ing.CHANNELS:
                v = ds.createVariable(channel, "f4", ("time", "lat", "lon"))
                v[:] = np.full((1, 4, 4), 1.0, dtype="f4")

    _, meta = ing.ingest_ncmrwf_cycle(str(test_file))
    lat_meta = meta["grid"]["lat"]
    # The masked cell must be excluded, not reported as a 9.97e36 fill value:
    assert lat_meta["count"] == 3
    assert 0.0 < lat_meta["min"] < 90.0
    assert 0.0 < lat_meta["max"] < 90.0
    # [10.0, 15.0, 25.0] diffs are [5.0, 10.0] -> mean step is 7.5:
    assert lat_meta["step"] == pytest.approx(7.5, abs=1e-3)
