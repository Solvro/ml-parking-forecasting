import pandas as pd
import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin

class ParkingImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values in parking data using a combination of forward-fill, seasonality-based filling, and median imputation.
    It also creates binary flag columns to indicate which values were imputed.
        Parameters:
        - cols_to_impute: A list of column names to impute (default is ['spaces_left']).
        - freq_minutes: The frequency in minutes for determining the forward-fill limit (default is 5 minutes).
        - ffill_limit: The maximum time in minutes to forward-fill missing values (default is 60 minutes).
        - convert_to_32: A boolean indicating whether to convert certain columns to 32-bit types to save memory (default is False).
        - copy: A boolean indicating whether to create a copy of the input DataFrame before transformation (default is True).
        - group_cols: A list of column names to group by before imputation (default is ['parking_id']).
    """
    def __init__(self, cols_to_impute=['spaces_left'], freq_minutes=5, ffill_limit=60, convert_to_32=False, copy=True, group_cols=['parking_id']):
        self.cols_to_impute = cols_to_impute
        self.freq_minutes = freq_minutes
        self.ffill_limit = ffill_limit
        self.convert_to_32 = convert_to_32
        self.copy = copy
        self.group_cols = group_cols

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("This transformer requires X to be a pandas DataFrame, not a numpy array.")
        self.group_fill_values_ = X.groupby(self.group_cols)[self.cols_to_impute].median()

        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns
        return self  

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("This transformer requires X to be a pandas DataFrame, not a numpy array.")
        if self.copy:
            X = X.copy()
        if not X.index.is_monotonic_increasing:
            X = X.sort_index()

        for col in self.cols_to_impute:
            X[f'{col}_is_imputed'] = X[col].isna().astype(int)

        X_grouped = X.groupby(self.group_cols)
        #Linear Interpolation leaks data ffill does not
        limit_small = self.ffill_limit // self.freq_minutes

        X[self.cols_to_impute] = X_grouped[self.cols_to_impute].ffill(limit=limit_small)

        # Seasonality Fill (7 days ago)
        records_per_week = 7 * 24 * (60 // self.freq_minutes)
        X[self.cols_to_impute] = X[self.cols_to_impute].fillna(
            X_grouped[self.cols_to_impute].shift(records_per_week)
        )

        # Seasonality Fill (28 days ago - 4 weeks)
        records_per_month = 4 * 7 * 24 * (60 // self.freq_minutes)
        X[self.cols_to_impute] = X[self.cols_to_impute].fillna(
            X_grouped[self.cols_to_impute].shift(records_per_month)
        )

        for col in self.cols_to_impute:
            # Map the specific column's median from the fit step
            fill_values = X[self.group_cols[0]].map(self.group_fill_values_[col])
            X[col] = X[col].fillna(fill_values)

        if self.convert_to_32:
            try:
                imputed_flags = [f'{c}_is_imputed' for c in self.cols_to_impute]
                target_cols = self.cols_to_impute + imputed_flags

                if 'active_users' in X.columns:
                    X['active_users'] = X['active_users'].astype(np.int32) # convert it back to int as active user should no longer have NaN values

                if 'spaces_left' in X.columns:
                    X['spaces_left'] = X['spaces_left'].astype(np.int32) # convert spaces left to int as it should no longer have NaN values after imputation

                float_cols = X[target_cols].select_dtypes(include=['float64']).columns
                int_cols = X[target_cols].select_dtypes(include=['int64']).columns
                X[float_cols] = X[float_cols].astype(np.float32)
                X[int_cols] = X[int_cols].astype(np.int32)
            except Exception as e:
                print(f"Error occurred while converting data types: {e}")

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_

        # Add the new flag columns to the output list
        new_flags = [f'{col}_is_imputed' for col in self.cols_to_impute]
        return list(input_features) + new_flags
