"""
Smoke-test every external data source credential needed for Phase 1.

Run this after filling in .env, before writing any real ingestion code, to
confirm each account/key actually works (Roadmap Milestone 0.2).

Usage:
    python scripts/smoke_test_sources.py
"""

import sys

import requests

from config.settings import get_settings

settings = get_settings()


def check_hurdat2() -> bool:
    """HURDAT2 is a public static file — no API key required, just reachability."""
    url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"
    try:
        resp = requests.head(url, timeout=10)
        ok = resp.status_code in (200, 301, 302)
        print(f"[HURDAT2]      {'OK' if ok else 'FAIL'} (status {resp.status_code})")
        return ok
    except requests.RequestException as e:
        print(f"[HURDAT2]      FAIL ({e})")
        return False


def check_cds() -> bool:
    if not settings.cds_api_key:
        print("[CDS/ERA5]     SKIPPED (CDS_API_KEY not set in .env yet)")
        return True  # not a failure at Phase 0 — just not configured yet
    try:
        resp = requests.get(
            f"{settings.cds_api_url}/v2",
            headers={"PRIVATE-TOKEN": settings.cds_api_key},
            timeout=10,
        )
        ok = resp.status_code < 500
        print(f"[CDS/ERA5]     {'OK' if ok else 'FAIL'} (status {resp.status_code})")
        return ok
    except requests.RequestException as e:
        print(f"[CDS/ERA5]     FAIL ({e})")
        return False


def check_firms() -> bool:
    if not settings.nasa_firms_map_key:
        print("[NASA FIRMS]   SKIPPED (NASA_FIRMS_MAP_KEY not set in .env yet)")
        return True
    try:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{settings.nasa_firms_map_key}/VIIRS_SNPP_NRT"
        resp = requests.get(url, timeout=10)
        ok = resp.status_code == 200
        print(f"[NASA FIRMS]   {'OK' if ok else 'FAIL'} (status {resp.status_code})")
        return ok
    except requests.RequestException as e:
        print(f"[NASA FIRMS]   FAIL ({e})")
        return False


def check_fred() -> bool:
    if not settings.fred_api_key:
        print("[FRED]         SKIPPED (FRED_API_KEY not set in .env yet)")
        return True
    try:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series",
            params={
                "series_id": "CPIAUCSL",
                "api_key": settings.fred_api_key,
                "file_type": "json",
            },
            timeout=10,
        )
        ok = resp.status_code == 200
        print(f"[FRED]         {'OK' if ok else 'FAIL'} (status {resp.status_code})")
        return ok
    except requests.RequestException as e:
        print(f"[FRED]         FAIL ({e})")
        return False


def main() -> None:
    print("Running ClimateGuard AI data-source smoke tests...\n")
    results = [check_hurdat2(), check_cds(), check_firms(), check_fred()]
    print()
    if all(results):
        print("All checks passed (or were skipped pending .env keys).")
        sys.exit(0)
    else:
        print("One or more checks failed — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
