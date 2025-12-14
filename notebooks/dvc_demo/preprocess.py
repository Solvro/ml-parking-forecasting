import pandas as pd
from pathlib import Path
# import importlib
# importlib.reload(utils)

PREPROC_DIR = Path("data/preproc")
PREPROC_DIR.mkdir(parents=True, exist_ok=True)

parking_av = pd.read_parquet('data/raw/parking_availabilities.parquet')
parkings = pd.read_parquet('data/raw/parkings.parquet')

parkings_slice = parkings.rename(columns={'id': 'parking_id'})[['parking_id', 'places']]
df = parking_av.merge(parkings_slice, on='parking_id', how='left')

df['spaces_left_pct'] = df['spaces_left'] / df['places']
df.to_parquet(PREPROC_DIR / "parking_df.parquet")
