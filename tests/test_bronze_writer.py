"""
Unit tests for data_pipeline.ingestion.common.bronze_writer.

Per Roadmap Task 1.1.2 acceptance criteria: "writes are idempotent
(re-running does not duplicate rows); each write is timestamped for lineage."
"""

import pandas as pd
import pytest

from data_pipeline.ingestion.common.bronze_writer import write_bronze


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "storm_id": ["AL011999", "AL011999"],
            "timestamp": pd.to_datetime(["1999-08-01", "1999-08-02"]),
            "max_wind_kt": [30, 40],
        }
    )


def test_write_bronze_creates_file(tmp_path, monkeypatch, sample_df):
    monkeypatch.chdir(tmp_path)
    out_path = write_bronze(sample_df, source="hurdat2", dataset="tracks")
    assert out_path.exists()

    written = pd.read_parquet(out_path)
    assert len(written) == 2
    assert "_ingested_at" in written.columns


def test_write_bronze_is_idempotent(tmp_path, monkeypatch, sample_df):
    """Writing the same data twice should not duplicate rows."""
    monkeypatch.chdir(tmp_path)
    write_bronze(
        sample_df,
        source="hurdat2",
        dataset="tracks",
        dedupe_keys=["storm_id", "timestamp"],
    )
    out_path = write_bronze(
        sample_df,
        source="hurdat2",
        dataset="tracks",
        dedupe_keys=["storm_id", "timestamp"],
    )

    written = pd.read_parquet(out_path)
    assert len(written) == 2  # not 4


def test_write_bronze_appends_new_rows(tmp_path, monkeypatch, sample_df):
    monkeypatch.chdir(tmp_path)
    write_bronze(
        sample_df,
        source="hurdat2",
        dataset="tracks",
        dedupe_keys=["storm_id", "timestamp"],
    )

    new_row = pd.DataFrame(
        {
            "storm_id": ["AL021999"],
            "timestamp": pd.to_datetime(["1999-09-15"]),
            "max_wind_kt": [45],
        }
    )
    out_path = write_bronze(
        new_row,
        source="hurdat2",
        dataset="tracks",
        dedupe_keys=["storm_id", "timestamp"],
    )

    written = pd.read_parquet(out_path)
    assert len(written) == 3


def test_write_bronze_rejects_empty_dataframe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Refusing to write an empty dataframe"):
        write_bronze(pd.DataFrame(), source="hurdat2", dataset="tracks")
