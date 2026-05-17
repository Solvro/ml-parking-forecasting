import pandas as pd
import numpy as np 
import datetime as dt
from sklearn.base import BaseEstimator, TransformerMixin
class ParkingSpacesCorrector(BaseEstimator, TransformerMixin):
    """Corrects parking space counts based on expansion status and calculates additional features related to capacity and utilization.
    First grouping columns should be parking_id as it is used for the mapping of expansion dates and previous capacities."""

    def __init__(self, parkings_df, expansion_date_map, previous_size_map, convert_to_32=False, copy = True, group_cols=['parking_id']):
        self.parkings_df = parkings_df
        self.expansion_date_map = expansion_date_map
        self.previous_size_map = previous_size_map
        self.convert_to_32 = convert_to_32
        self.copy = copy
        self.group_cols = group_cols
        if not isinstance(parkings_df, pd.DataFrame):
            raise TypeError("This transformer requires parkings_df to be a pandas DataFrame, not a numpy array.")

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("This transformer requires X to be a pandas DataFrame, not a numpy array.")
        if hasattr(X, "columns"):
                self.feature_names_in_ = X.columns

        # Pre-calculate mapping data once during fit
        # If parkings_df doesn't ship `max_spaces_left`, derive it from training data.
        # Doing it here (in fit) — not in __init__ — keeps the value learned only on X
        # (the train fold), which is the only leakage-safe place to compute it.
        parkings = self.parkings_df
        if 'max_spaces_left' not in parkings.columns:
            missing = {'id'}.difference(parkings.columns)
            if missing:
                raise KeyError(
                    f"parkings_df must contain 'id' to derive 'max_spaces_left' from X. Missing: {sorted(missing)}."
                )
            observed_max = X.groupby('parking_id')['spaces_left'].max()
            parkings = parkings.merge(
                observed_max.rename('max_spaces_left'),
                left_on='id', right_index=True, how='left',
            )
        meta = parkings[['id', 'name', 'max_spaces_left']].copy()

        # Map expansion dates and previous capacities
        meta['exp_date'] = meta['name'].map(self.expansion_date_map).fillna("2200-01-01")
        meta['cap_before'] = meta['name'].map(self.previous_size_map).fillna(meta['max_spaces_left'])

        # Create a fast lookup dataframe indexed by 'id' 
        self.parkings_meta_ = meta.set_index('id')[['exp_date', 'cap_before', 'max_spaces_left']]
        return self  

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("This transformer requires X to be a pandas DataFrame, not a numpy array.")
        if self.copy:
            X = X.copy()

        X['overbooked'] = (X['spaces_left'] < 0).astype(int)
        X['overbooked_spaces'] = np.where(X['overbooked'] == 1, -X['spaces_left'], 0)

        X = X.join(self.parkings_meta_, on=self.group_cols[0])

        X['is_expanded'] = (X['measured_at'] >= X['exp_date']).astype(int)

        # Capacity assignment
        X['total_capacity'] = np.where(X['is_expanded'] == 1, X['max_spaces_left'], X['cap_before'])

        # Clip spaces_left to be between 0 and total_capacity
        X['spaces_left'] = X['spaces_left'].clip(lower=0, upper=X['total_capacity'])

        # Calculate utilization rate
        X['utilization_rate'] = (X['total_capacity'] - X['spaces_left']) / X['total_capacity']

        # Clean up temporary columns
        X = X.drop(columns=['exp_date', 'cap_before', 'max_spaces_left'])

        if self.convert_to_32:
            try:
                int_cols = ['overbooked', 'is_expanded', 'overbooked_spaces', 'total_capacity']
                float_cols = ['utilization_rate']
                X[int_cols] = X[int_cols].astype(np.int32)
                X[float_cols] = X[float_cols].astype(np.float32)
            except Exception as e:
                print(f"Error occurred while converting data types: {e}")

        return X
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return list(input_features) + ['overbooked', 'overbooked_spaces', 'is_expanded', 'total_capacity', 'utilization_rate']
