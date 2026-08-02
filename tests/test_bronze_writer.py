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

def test_write_bronze_output_is_spark_readable(tmp_path, monkeypatch, sample_df):
    """
    Regression test for a real bug hit while building Phase 3: pandas/
    pyarrow default to nanosecond-precision Parquet timestamps, which
    Spark's built-in Parquet reader cannot read at all (raises
    'Illegal Parquet type: INT64 (TIMESTAMP(NANOS,false))'). This test
    actually round-trips a Bronze write through Spark, so a future change
    that reintroduces nanosecond-precision timestamps would fail here
    immediately rather than being discovered downstream in Phase 3.
    """
    import pyarrow.parquet as pq

    monkeypatch.chdir(tmp_path)
    out_path = write_bronze(sample_df, source="hurdat2", dataset="tracks")

    # Cheap check first: confirm the physical Parquet type is NOT nanoseconds.
    schema = pq.read_schema(out_path)
    timestamp_field = schema.field("timestamp")
    assert timestamp_field.type.unit != "ns", (
        "Bronze writer regressed to nanosecond-precision timestamps — "
        "this will break every downstream Spark read (Phase 3+)."
    )

    # Real check: an actual Spark session can read the file without error.
    from data_pipeline.databricks_jobs.geo_join_era5 import get_spark_session

    spark = get_spark_session(app_name="test-bronze-spark-readability")
    try:
        # IMPORTANT: pass an ABSOLUTE path. The Spark JVM subprocess is
        # long-lived (reused across test modules via getOrCreate()) and
        # does not track Python's monkeypatch.chdir() — a relative path
        # here would resolve against the JVM's original working directory,
        # not this test's tmp_path, and fail with PATH_NOT_FOUND.
        sdf = spark.read.parquet(str(out_path.resolve()))
        assert sdf.count() == 2
    finally:
        spark.stop()