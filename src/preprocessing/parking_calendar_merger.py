import pandas as pd
import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin

class ParkingCalendarMerger(BaseEstimator, TransformerMixin):
    """Merges parking data with calendar data based on nearest dates and adds a day type feature."""

    def __init__(self, calendar_df, freq_minutes=5, convert_to_32=False, copy = True):
        self.calendar_df = calendar_df[['date', 'day_type_id']].copy()
        self.freq_minutes = freq_minutes
        self.convert_to_32 = convert_to_32
        self.copy = copy

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns
        return self  

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("This transformer requires X to be a pandas DataFrame, not a numpy array.")
        if self.copy:
            X = X.copy()
        #check if columns required to merge exist
        try: 
            parking_subset = X[['measured_at']]
        except KeyError as e:
            raise KeyError(f"Missing required columns in input X. Expected 'measured_at'. Original error: {e}")

        X['measured_at'] = pd.to_datetime(X['measured_at'])
        X = X.sort_values('measured_at')

        X['merge_date'] = X['measured_at'].dt.date
        #check if calendar_df has the required columns for merging
        try :
            calendar_subset = self.calendar_df[['date']]
        except KeyError as e:
            raise KeyError(f"Missing required columns in calendar_df. Expected 'date'. Original error: {e}")

        X = pd.merge(X, self.calendar_df, left_on='merge_date', right_on='date', how='left')

        X = X.drop(columns=['date', 'merge_date'])  # Drop the redundant 'date' and 'merge_date' columns after merge

        if self.convert_to_32:
            X['day_type_id'] = X['day_type_id'].astype(np.int32)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return list(input_features) + ['day_type_id']
