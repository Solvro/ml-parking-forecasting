import pandas as pd
import numpy as np 
import datetime as dt
from sklearn.base import BaseEstimator, TransformerMixin
class ParkingResampler(BaseEstimator, TransformerMixin):
    """Custom transformer for resampling parking availability data.
    It takes a Dataframe and returns a resampled version of it based on the specified aggregation rules and frequency.
        Parameters:
        - agg_rules: A dictionary specifying the aggregation rules for each column (e.g., {'spaces
        _left': 'mean'}).
        - rule: A string representing the resampling frequency (default is '5min').
        - convert_to_32: A boolean indicating whether to convert integer and float columns to 32
        bit types to save memory (default is False).
        - copy: A boolean indicating whether to create a copy of the input DataFrame before transformation (default is True).
        - group_cols: A list of column names to group by before resampling (default is ['parking_id']).
    """
    def __init__(self, agg_rules, rule='5min', convert_to_32=False, copy = True, group_cols =['parking_id']):
        self.rule = rule
        self.convert_to_32 = convert_to_32
        self.copy = copy
        self.group_cols = group_cols
        self.agg_rules = agg_rules


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("This transformer requires X to be a pandas DataFrame, not a numpy array.")
        # Ensure measured_at is the index for resampling
        if self.copy:
            X = X.copy()

        grouping_instructions = self.group_cols + [pd.Grouper(key='measured_at', freq=self.rule)]

        # Perform the grouping and aggregation in one go
        df_resampled = (
            X.groupby(grouping_instructions)
             .agg(self.agg_rules)
             .reset_index()
        )

        if self.convert_to_32:
            try:
                int_columns = df_resampled.select_dtypes(include=['int64']).columns
                df_resampled[int_columns] = df_resampled[int_columns].astype(np.int32)
                float_columns = df_resampled.select_dtypes(include=['float64']).columns
                df_resampled[float_columns] = df_resampled[float_columns].astype(np.float32)
            except Exception as e:
                print(f"Error occurred while converting data types: {e}")   
        return df_resampled
