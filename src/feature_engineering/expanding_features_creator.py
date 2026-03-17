import pandas as pd
import numpy as np 
import datetime as dt
from sklearn.base import BaseEstimator, TransformerMixin
class ExpandingFeaturesCreator(BaseEstimator, TransformerMixin):
    """Creates expanding features such as cumulative mean, max, min, and trend for specified columns based on parking_id groups."""
    def __init__(self, expanding_features, convert_to_32=False, shift_guard_periods=1, copy=True, group_cols=['parking_id'], fill_nan_with_zero=False):
        self.expanding_features = expanding_features
        self.group_cols = group_cols
        self.convert_to_32 = convert_to_32
        self.copy = copy
        self.shift_guard_periods = shift_guard_periods
        self.fill_nan_with_zero = fill_nan_with_zero

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns

        return self  

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("This transformer requires X to be a pandas DataFrame, not a numpy array.")
        if self.copy:
            X = X.copy()

        for _, (suffix, feature_type, columns) in self.expanding_features.items():
            for col in columns:
                new_col_name = f"{col}_{suffix}"


                X[new_col_name] = (
                    X.groupby(self.group_cols)[col]
                     .expanding()
                     .agg(feature_type)   # Uses "mean", "max", "min", "std" dynamically
                     .groupby(level=0) 
                     .shift(self.shift_guard_periods)
                     .reset_index(level=0, drop=True)
                )
                if self.fill_nan_with_zero:
                    X[new_col_name] = X[new_col_name].fillna(0)

        if self.convert_to_32:
            try:    
                new_cols = [f"{col}_{suffix}" for _, (suffix, _, columns) in self.expanding_features.items() for col in columns]
                int_cols = X[new_cols].select_dtypes(include=['int64']).columns
                float_cols = X[new_cols].select_dtypes(include=['float64']).columns
                X[int_cols] = X[int_cols].astype(np.int32)
                X[float_cols] = X[float_cols].astype(np.float32)
            except Exception as e:
                print(f"Error occurred while converting data types: {e}")
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_

        new_feature_names = []
        for _, (suffix, _, columns) in self.expanding_features.items():
            for col in columns:
                new_feature_names.append(f"{col}_{suffix}")

        return list(input_features) + new_feature_names
