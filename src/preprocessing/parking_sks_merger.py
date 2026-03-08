import pandas as pd
import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin

class ParkingSKSMerger(BaseEstimator, TransformerMixin):
    """Merges parking data with SKS user data based on nearest timestamps and calculates a ratio feature. Data from SKS 
    is shifted by a specified lag to account for data propagation delays and to prevent data leakage."""

    def __init__(self, sks_df, parkings_df, freq_minutes=5, tolerance=10, convert_to_32=False, users_lag=12, copy = True):
        self.sks_df = sks_df[['external_timestamp', 'active_users']].sort_values('external_timestamp')
        self.parkings_df = parkings_df[['id', 'distance_to_sks']].copy()
        self.freq_minutes = freq_minutes
        self.tolerance = tolerance
        self.convert_to_32 = convert_to_32
        self.users_lag = users_lag
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

        X = X.sort_values('measured_at')
        #check if columns required to merge exist 
        try:
            parkings_subset = self.parkings_df[['id']]
        except KeyError as e:
            raise KeyError(f"Missing required columns in parkings_df. {e}")

        try:
            parking_subset = X[['parking_id', 'measured_at']]
        except KeyError as e:
            raise KeyError(f"Missing required columns in input X. Expected 'parking_id' and 'measured_at'. Original error: {e}")


        X = pd.merge(X, self.parkings_df, left_on='parking_id', right_on='id', how='left')
        X = X.drop(columns=['id'])  # Drop the redundant 'id' column after merge

        sks_lagged = self.sks_df.copy()
        sks_lagged['active_users'] = sks_lagged['active_users'].shift(self.users_lag)

        try:
            sks_subset = sks_lagged[['external_timestamp']]
        except KeyError as e:
            raise KeyError(f"Missing required columns in SKS data. {e}")

        X = pd.merge_asof(
            X, 
            sks_lagged, 
            left_on='measured_at', 
            right_on='external_timestamp', 
            direction='nearest', 
            tolerance=pd.Timedelta(minutes=self.tolerance)
        )

        X['sks_dist_to_users_ratio'] = round(X['active_users'] * (100/(X['distance_to_sks'] + 1e-6)), 2)  # Add small constant to avoid division by zero

        # Drop the extra timestamp column 
        if 'external_timestamp' in X.columns:
            X = X.drop(columns=['external_timestamp'])

        if self.convert_to_32:
            try:
                X['active_users'] = X['active_users'].astype(np.float32) # temporary conver it to float 32 because float allows for NaN values which we have in this column due to the merge_asof and the tolerance"
                X['distance_to_sks'] = X['distance_to_sks'].astype(np.float32)
                X['sks_dist_to_users_ratio'] = X['sks_dist_to_users_ratio'].astype(np.float32)
            except Exception as e:
                print(f"Error occurred while converting data types: {e}")
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return list(input_features) + ['active_users', 'distance_to_sks', 'sks_dist_to_users_ratio']
