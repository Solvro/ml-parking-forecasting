import pandas as pd
import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin

class CannibalisationFeaturesCreator(BaseEstimator, TransformerMixin):
    """Adds cannibalisation features based on the nearest neighboring parkings' data, with a guard period to prevent data leakage."""
    def __init__(self, parkings_df, cannibalisation_features, cannibalisation_guard=12, convert_to_32=False, copy=True, group_cols=['parking_id'], fill_nan_with_zero=False):
        self.parkings_df = parkings_df.copy()
        self.cannibalisation_features = cannibalisation_features
        self.cannibalisation_guard = cannibalisation_guard
        self.convert_to_32 = convert_to_32
        self.copy = copy
        self.fill_nan_with_zero = fill_nan_with_zero
        self.group_cols = group_cols

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns
        return self   

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("This transformer requires X to be a pandas DataFrame, not a numpy array.")
        if self.copy:
            X_out = X.copy()
        else:
            X_out = X

        required_neighbor_map = {} 
        for _, (_, neighbor_type, _) in self.cannibalisation_features.items():
            col_name = f"{neighbor_type}_id"
            required_neighbor_map[neighbor_type] = col_name

        cols_to_merge = list(set(required_neighbor_map.values()))

        #check if columns required to merge exist 
        try:
            parkings_subset = self.parkings_df[['id'] + cols_to_merge]
        except KeyError as e:
            raise KeyError(f"Missing required columns in parkings_df. Expected 'id' and {cols_to_merge}. Original error: {e}")
        # Merge metadata (Left join to keep X structure)
        X_merged = X_out.merge(
            self.parkings_df[['id'] + cols_to_merge],
            left_on='parking_id',
            right_on='id',
            how='left'
        ).drop(columns=['id'])

        for key, (suffix, neighbor_type, columns) in self.cannibalisation_features.items():

            neighbor_col_name = required_neighbor_map.get(neighbor_type)

            if neighbor_col_name not in X_merged.columns:
                continue

            # Prepare the "Lookup" table (The neighbor data)
            lookup_df = X[['measured_at', 'parking_id'] + columns].copy()

            # Apply Cannibalisation Guard (Latency Simulation)
            if self.cannibalisation_guard <= 0:
                self.cannibalisation_guard = 12  # Default to 1 hour if invalid value is provided
                raise ValueError("Cannibalisation guard must be a positive integer representing the number of periods to shift.")

            if self.cannibalisation_guard > 0:
                lookup_df = lookup_df.sort_values('measured_at')
                lookup_df[columns] = lookup_df.groupby(self.group_cols)[columns].shift(self.cannibalisation_guard)

            # Rename columns for the merge
            rename_dict = {col: f"{col}_{suffix}" for col in columns}
            rename_dict['parking_id'] = neighbor_col_name

            lookup_df = lookup_df.rename(columns=rename_dict)


            # Merge
            X_merged = X_merged.merge(
                lookup_df,
                on=['measured_at', neighbor_col_name],
                how='left'
            )

            # Fill NaNs created by the shift or missing neighbors
            new_cols = [f"{col}_{suffix}" for col in columns]
            if self.fill_nan_with_zero:
                X_merged[new_cols] = X_merged[new_cols].fillna(0)

        X_merged = X_merged.drop(columns=cols_to_merge, errors='ignore')

        if self.convert_to_32:
            all_new_cols = []
            for _, (suffix, _, columns) in self.cannibalisation_features.items():
                all_new_cols.extend([f"{col}_{suffix}" for col in columns])
            try:
                cols_to_convert = [c for c in all_new_cols if c in X_merged.columns]
                int_cols = X_merged[cols_to_convert].select_dtypes(include=['int64']).columns
                float_cols = X_merged[cols_to_convert].select_dtypes(include=['float64']).columns
                X_merged[int_cols] = X_merged[int_cols].astype(np.int32)
                X_merged[float_cols] = X_merged[float_cols].astype(np.float32)
            except Exception as e:
                print(f"Error occurred while converting data types: {e}")   

        return X_merged

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_

        new_feature_names = []
        for _, (suffix, _, columns) in self.cannibalisation_features.items():
            for col in columns:
                new_feature_names.append(f"{col}_{suffix}")

        return list(input_features) + new_feature_names
