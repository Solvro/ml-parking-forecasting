import pandas as pd
import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin
"""Adds day-related features such as day of the week, weekend indicator, and parking open status based on parking hours."""
class DayFeaturesCreator(BaseEstimator, TransformerMixin):
    def __init__(self, df_parkings, convert_to_32=False):
        self.df_parkings = df_parkings.copy()
        self.convert_to_32 = convert_to_32

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
                self.feature_names_in_ = X.columns
        return self  

    def transform(self, X):
        X = X.copy()
        X['day_of_week'] = X['measured_at'].dt.dayofweek
        X['is_weekend'] = X['day_of_week'].isin([5, 6]).astype(int)

        parking_info = self.df_parkings[['id', 'open_hour', 'close_hour']]

        #Temporary merge to get open/close hours for each parking_id
        X_merged = X.merge(parking_info, left_on='parking_id', right_on='id', how='left')

        current_time_str = X['measured_at'].dt.strftime('%H:%M:%S')

        # Ensure start/end are strings for comparison
        open_time = X_merged['open_hour'].astype(str)
        close_time = X_merged['close_hour'].astype(str)

        # 4. "Is Open" Logic
        is_open_standard = (current_time_str >= open_time) & (current_time_str <= close_time)

        # Always open logic
        always_open = (open_time == '00:00:00') & (close_time == '00:00:00')

        # Combine logic: Open if standard rule applies OR if it's a 24/7 lot
        X['is_open'] = (is_open_standard | always_open).astype(int)

        # Fill NaNs with 0 (closed) just in case a parking_id didn't match
        X['is_open'] = X['is_open'].fillna(0).astype(int)

        if self.convert_to_32:
            numerical_cols = ['day_of_week', 'is_weekend', 'is_open']
            X[numerical_cols] = X[numerical_cols].astype(np.int32)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return list(input_features) + ['day_of_week', 'is_weekend', 'is_open']
