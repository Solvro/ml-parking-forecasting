import pandas as pd
import numpy as np 
import datetime as dt
from sklearn.base import BaseEstimator, TransformerMixin
class ParkingSpacesCorrector(BaseEstimator, TransformerMixin):
    """Corrects parking space counts based on expansion status and calculates additional features related to capacity and utilization."""

    def __init__(self, parkings_df, expansion_date_map, previous_size_map, convert_to_32=False):
        self.parkings_df = parkings_df
        self.expansion_date_map = expansion_date_map
        self.previous_size_map = previous_size_map
        self.convert_to_32 = convert_to_32

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
                self.feature_names_in_ = X.columns
        return self  

    def transform(self, X):
        X = X.copy()

        X['overbooked'] = (X['spaces_left'] < 0).astype(int)
        X['overbooked_spaces'] = np.where(X['overbooked'] == 1, -X['spaces_left'], 0)

        X_grouped = X.groupby('parking_id')

        for parking_id, group in X_grouped:
            # Determine expansion status based on parking_id and measured_at
            parking_data = self.parkings_df[self.parkings_df['id'] == parking_id].iloc[0]
            exp_date_str = self.expansion_date_map.get(parking_data['name'], "2200-01-01")
            group['is_expanded'] = (group['measured_at'] >= exp_date_str).astype(int)
            X.loc[group.index, 'is_expanded'] = group['is_expanded']

            #cqalculate total capacity based on expansion status
            cap_after_expansion = parking_data['max_spaces_left']
            cap_before_expansion = self.previous_size_map.get(parking_data['name'], cap_after_expansion)
            group['total_capacity'] = np.where(group['is_expanded'] == 1, cap_after_expansion, cap_before_expansion)
            X.loc[group.index, 'total_capacity'] = group['total_capacity']

            # Ensure spaces_left does not exceed total_capacity and is not negative
            group['spaces_left'] = group['spaces_left'].clip(upper=group['total_capacity'])
            group['spaces_left'] = group['spaces_left'].clip(lower=0)
            X.loc[group.index, 'spaces_left'] = group['spaces_left']

            # Calculate utilization rate
            X.loc[group.index, 'utilization_rate'] = (X['total_capacity'] - X['spaces_left']) / X['total_capacity']


        if self.convert_to_32:
            int_cols = ['overbooked', 'is_expanded', 'overbooked_spaces', 'total_capacity']
            float_cols = ['utilization_rate']
            X[int_cols] = X[int_cols].astype(np.int32)
            X[float_cols] = X[float_cols].astype(np.float32)
        return X
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return list(input_features) + ['overbooked', 'overbooked_spaces', 'is_expanded', 'total_capacity', 'utilization_rate']
