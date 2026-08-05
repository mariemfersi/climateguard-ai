import pandas as pd
import pyarrow.dataset as ds

# Check which years have real (non-ffilled) SST vs filled
era5_years = sorted(pd.to_datetime(
    ds.dataset("data_pipeline/bronze/era5/gulf_coast_reanalysis.parquet", format="parquet")
    .to_table(columns=["timestamp"]).to_pandas()["timestamp"]
).dt.year.unique())

all_years = set(range(1950, 2024))
missing_years = sorted(all_years - set(era5_years))

print(f"ERA5 coverage years: {len(era5_years)} ({min(era5_years)}-{max(era5_years)})")
print(f"Missing years: {missing_years}")
print(f"Missing count: {len(missing_years)}")
