import pandas as pd
import pyarrow.dataset as ds

era5_path = "data_pipeline/bronze/era5/gulf_coast_reanalysis.parquet"

# read timestamps from parquet via pyarrow dataset to avoid loading full file
dataset = ds.dataset(era5_path, format="parquet")

table = dataset.to_table(columns=["timestamp"]) 
# convert to pandas datetime and extract years
years = pd.to_datetime(table.to_pandas()["timestamp"]).dt.year.unique()
years = sorted(int(y) for y in years)
print("ERA5 years present:", years)

all_years = set(range(1950, 2024))
missing = sorted(list(all_years - set(years)))
print("Missing years:", missing)
print("Count present:", len(years))
