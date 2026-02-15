import pandas as pd
import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin

class ParkingCalendarMerger(BaseEstimator, TransformerMixin):
    """Merges parking data with calendar data based on nearest dates and adds a day type feature."""

    def __init__(self, calendar_df, freq_minutes=5, tolerance=10, convert_to_32=False):
        self.calendar_df = calendar_df[['date', 'day_type_id']].copy()
        self.freq_minutes = freq_minutes
        self.tolerance = tolerance
        self.convert_to_32 = convert_to_32

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns
        return self  

    def transform(self, X):
        X = X.copy()
        X['measured_at'] = pd.to_datetime(X['measured_at'])
        X = X.sort_values('measured_at')

        X['merge_date'] = X['measured_at'].dt.date
        X = pd.merge(X, self.calendar_df, left_on='merge_date', right_on='date', how='left')

        X = X.drop(columns=['date', 'merge_date'])  # Drop the redundant 'date' and 'merge_date' columns after merge

        if self.convert_to_32:
            X['day_type_id'] = X['day_type_id'].astype(np.int32)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return list(input_features) + ['day_type_id']
