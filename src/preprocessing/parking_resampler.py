import pandas as pd
import numpy as np 
import datetime as dt
from sklearn.base import BaseEstimator, TransformerMixin
class ParkingResampler(BaseEstimator, TransformerMixin):
    def __init__(self, rule='5min', convert_to_32=False):
        self.rule = rule
        self.convert_to_32 = convert_to_32
        self.agg_rules = {
            'spaces_left': 'mean',
            #'trend': 'mean', # this trend is weird.. i still have no idea how to use it seems semi random at times 
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure measured_at is the index for resampling
        X = X.copy()
        X = X.sort_values(by=['measured_at'])
        X = X.reset_index(drop=True)
        X = X.set_index('measured_at')

        # Group by parking_id and resample each group
        df_resampled = (
            X.groupby('parking_id')
             .resample(self.rule)
             .agg(self.agg_rules)
             .reset_index()
        )
        if self.convert_to_32:
            int_columns = df_resampled.select_dtypes(include=['int64']).columns
            df_resampled[int_columns] = df_resampled[int_columns].astype(np.int32)
            float_columns = df_resampled.select_dtypes(include=['float64']).columns
            df_resampled[float_columns] = df_resampled[float_columns].astype(np.float32)
        return df_resampled
