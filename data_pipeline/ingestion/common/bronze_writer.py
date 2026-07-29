"""
Shared Bronze-layer write utility.

Every ingestion source (HURDAT2, ERA5, FIRMS, FRED) writes its raw data
through this single function so that:
  1. Writes are idempotent — re-running an ingestion job does not duplicate rows.
  2. Every row is stamped with an ingestion timestamp for lineage/debugging.
  3. The Bronze layer has one consistent on-disk convention, making the later
     migration to ADLS Gen2 (Phase 3) a path-prefix change, not a rewrite.

Local dev writes to ./data_pipeline/bronze/<source>/... ; Phase 3 swaps the
base path for an ADLS Gen2 URI via config.settings — the function signature
does not change.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

BRONZE_ROOT = Path("data_pipeline/bronze")


def write_bronze(
    df: pd.DataFrame,
    source: str,
    dataset: str,
    dedupe_keys: list[str] | None = None,
) -> Path:
    """
    Write a dataframe to the Bronze layer for a given source/dataset,
    stamping an ingestion timestamp and de-duplicating against any
    previously written data for the same dataset.

    Args:
        df: raw dataframe to write (schema is the source's responsibility).
        source: e.g. "hurdat2", "era5", "firms", "fred".
        dataset: e.g. "tracks", "sst_gulf_coast" — allows a source to have
            more than one logical dataset.
        dedupe_keys: columns that uniquely identify a row, used to drop
            duplicates introduced by re-running the ingestion job. If None,
            full-row duplicates are dropped instead.

    Returns:
        Path to the written parquet file.

    Raises:
        ValueError: if df is empty — writing an empty Bronze file silently
            would hide a broken upstream fetch.
    """
    if df.empty:
        raise ValueError(
            f"Refusing to write an empty dataframe to Bronze for "
            f"source='{source}', dataset='{dataset}'. This almost always "
            f"means the upstream fetch failed silently — check the fetcher, "
            f"not this writer."
        )

    out_dir = BRONZE_ROOT / source
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}.parquet"

    df = df.copy()
    df["_ingested_at"] = dt.datetime.now(dt.UTC).isoformat()

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, df], ignore_index=True)
        keys = dedupe_keys or list(df.columns.drop("_ingested_at"))
        combined = combined.drop_duplicates(subset=keys, keep="last")
    else:
        combined = df

    combined.to_parquet(out_path, index=False)
    return out_path
