import pandas as pd
from pathlib import Path
from utils import make_lag, make_rolling, make_expanding
# import importlib
# importlib.reload(utils)

FEATURE_ENG_DIR = Path("data/feature_eng")
FEATURE_ENG_DIR.mkdir(parents=True, exist_ok=True)

parking_df = pd.read_parquet('data/preproc/parking_df.parquet')

df_lag = make_lag(parking_df, core_column='spaces_left', lag= [1, 7, 14], group_cols='parking_id')
df_lag.to_parquet(FEATURE_ENG_DIR / "parking_df_lag.parquet")
