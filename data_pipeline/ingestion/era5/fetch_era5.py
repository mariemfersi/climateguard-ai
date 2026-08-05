"""
Fetch ERA5 reanalysis data (sea-surface temperature, mean sea-level
pressure, 10m wind components) from the Copernicus Climate Data Store,
scoped to the Gulf Coast / Atlantic basin bounding box.

Deliberately NOT global-extent: per Roadmap Phase 1 risk notes, requesting
global ERA5 data wastes days of download time and disk space for data the
vertical slice (Florida/Gulf Coast hurricane) will never use.

Usage:
    python -m data_pipeline.ingestion.era5.fetch_era5
"""

from __future__ import annotations

import logging
from pathlib import Path


from config.settings import get_settings

logger = logging.getLogger(__name__)


# Expose a module-level `cdsapi` symbol so unit tests can patch
# `data_pipeline.ingestion.era5.fetch_era5.cdsapi.Client`. If the
# real `cdsapi` package is present it will be imported below at
# runtime inside `fetch_era5_raw`; otherwise a lightweight dummy
# placeholder is provided so `mock.patch` can target the attribute.
try:  # pragma: no cover - environment dependent
    import cdsapi as cdsapi  # type: ignore
except Exception:  # pragma: no cover - allow tests to patch this
    class _CdsApiPlaceholder:
        class Client:  # simple placeholder class
            def __init__(self, *a, **k):
                raise ImportError("cdsapi not installed")

    cdsapi = _CdsApiPlaceholder()

# [North, West, South, East] — covers the Gulf of Mexico, Florida, and the
# western Atlantic approach used by most Gulf Coast-landfalling hurricanes.
# Widen this only if a later phase expands beyond the Florida/Gulf Coast
# vertical slice (see design doc's horizontal-expansion note).
GULF_COAST_BBOX = [35, -100, 10, -60]

RAW_DOWNLOAD_PATH = Path("data_pipeline/bronze/era5/_raw_era5.nc")


def fetch_era5_raw(
    year: str,
    months: list[str],
    dest: Path = RAW_DOWNLOAD_PATH,
    area: list[float] = GULF_COAST_BBOX,
) -> Path:
    """
    Download ERA5 monthly-means-by-hour-of-day reanalysis fields for the
    given year/months, scoped to `area`.

    Args:
        year: e.g. "2023".
        months: e.g. ["06", "07", "08", "09", "10", "11"] (hurricane season).
        dest: output NetCDF path.
        area: [North, West, South, East] bounding box.

    Raises:
        RuntimeError: if CDS_API_KEY is not configured (see .env.example).
    """
    settings = get_settings()
    settings.require("cds_api_key")

    logger.info("Requesting ERA5 for year=%s months=%s area=%s", year, months, area)

    # Use the module-level `cdsapi` symbol (tests may have patched it).
    try:
        client = cdsapi.Client(url=settings.cds_api_url, key=settings.cds_api_key)
    except Exception as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "cdsapi is required to fetch ERA5 data; install via `pip install cdsapi`"
        ) from e

    dest.parent.mkdir(parents=True, exist_ok=True)

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "sea_surface_temperature",
                "mean_sea_level_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "year": year,
            "month": months,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": area,
            "format": "netcdf",
        },
        str(dest),
    )

    logger.info("Wrote raw ERA5 NetCDF to %s", dest)
    return dest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example: one recent hurricane season, adjust as needed per training run.
    fetch_era5_raw(year="2023", months=["06", "07", "08", "09", "10", "11"])
