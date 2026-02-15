import pandas as pd
import numpy as np 
import datetime as dt
from sklearn.base import BaseEstimator, TransformerMixin
class ExpandingFeaturesCreator(BaseEstimator, TransformerMixin):
    """Creates expanding features such as cumulative mean, max, min, and trend for specified columns based on parking_id groups."""
    def __init__(self, expanding_features, group_col='parking_id', convert_to_32=False, shift_guard_minutes=1):
        self.expanding_features = expanding_features
        self.group_col = group_col
        self.convert_to_32 = convert_to_32
        self.shift_guard_minutes = shift_guard_minutes

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns

        return self  

    def transform(self, X):
        X = X.copy()

        for _, (suffix, feature_type, columns) in self.expanding_features.items():
            for col in columns:
                new_col_name = f"{col}_{suffix}"


                X[new_col_name] = (
                    X.groupby(self.group_col)[col]
                     .expanding()
                     .agg(feature_type)   # Uses "mean", "max", "min", "std" dynamically
                     .shift(self.shift_guard_minutes)
                     .reset_index(level=0, drop=True)
                )

                X[new_col_name] = X[new_col_name].fillna(0)

        if self.convert_to_32:
            new_cols = [f"{col}_{suffix}" for _, (suffix, _, columns) in self.expanding_features.items() for col in columns]
            int_cols = X[new_cols].select_dtypes(include=['int64']).columns
            float_cols = X[new_cols].select_dtypes(include=['float64']).columns
            X[int_cols] = X[int_cols].astype(np.int32)
            X[float_cols] = X[float_cols].astype(np.float32)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_

        new_feature_names = []
        for _, (suffix, _, columns) in self.expanding_features.items():
            for col in columns:
                new_feature_names.append(f"{col}_{suffix}")

        return list(input_features) + new_feature_names
