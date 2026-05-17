import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Order in which neighbor types map to ranks (0 = closest).
_NEIGHBOR_RANK = {
    "closest": 0,
    "second_closest": 1,
    "third_closest": 2,
    "fourth_closest": 3,
}

EARTH_RADIUS_M = 6_371_000

def _haversine_distance_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two WGS84 points."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def _derive_neighbor_id_columns(parkings_df: pd.DataFrame, neighbor_id_cols: list[str]) -> pd.DataFrame:
    """For each parking, sort other parkings by haversine distance and assign neighbor IDs by rank.

    Returns a copy of parkings_df with the requested ``*_id`` columns added.
    Skips columns already present.
    """
    missing_cols = [c for c in neighbor_id_cols if c not in parkings_df.columns]
    if not missing_cols:
        return parkings_df.copy()

    geo_required = {'id', 'geo_lat', 'geo_lan'}.difference(parkings_df.columns)
    if geo_required:
        raise KeyError(
            f"parkings_df must contain {missing_cols} or ['id','geo_lat','geo_lan'] for haversine "
            f"computation. Missing geo columns: {sorted(geo_required)}."
        )

    rank_for = {f"{name}_id": _NEIGHBOR_RANK[name] for name in _NEIGHBOR_RANK}
    unknown = [c for c in missing_cols if c not in rank_for]
    if unknown:
        raise ValueError(
            f"Cannot derive {unknown}: only neighbor types {list(_NEIGHBOR_RANK)} are supported."
        )

    out = parkings_df.copy()
    ids   = out['id'].to_numpy()
    lats  = out['geo_lat'].to_numpy(dtype=float)
    lons  = out['geo_lan'].to_numpy(dtype=float)
    n = len(ids)

    # Pairwise haversine via broadcasting; n is tiny (~5), so this is fine.
    dist = _haversine_distance_m(lats[:, None], lons[:, None], lats[None, :], lons[None, :])
    np.fill_diagonal(dist, np.inf)  # exclude self
    order = np.argsort(dist, axis=1)  # each row: indices of other parkings, nearest first

    for col in missing_cols:
        rank = rank_for[col]
        if rank >= n - 1:
            raise ValueError(
                f"Cannot assign '{col}' (rank {rank}): only {n - 1} other parking(s) available."
            )
        out[col] = ids[order[:, rank]]

    return out


class CannibalisationFeaturesCreator(BaseEstimator, TransformerMixin):
    """Adds cannibalisation features based on the nearest neighboring parkings' data, with a guard period to prevent data leakage.

    Neighbor IDs (``closest_id``, ``second_closest_id``, ...) are taken from ``parkings_df`` if present;
    otherwise they're derived in __init__ via pairwise haversine distance from ``geo_lat``/``geo_lan``."""
    def __init__(self, parkings_df, cannibalisation_features, cannibalisation_guard=12, convert_to_32=False, copy=True, group_cols=['parking_id'], fill_nan_with_zero=False):
        self.cannibalisation_features = cannibalisation_features
        self.cannibalisation_guard = cannibalisation_guard
        self.convert_to_32 = convert_to_32
        self.copy = copy
        self.fill_nan_with_zero = fill_nan_with_zero
        self.group_cols = group_cols

        required_neighbor_id_cols = sorted({
            f"{neighbor_type}_id"
            for _, (_, neighbor_type, _) in cannibalisation_features.items()
        })
        self.parkings_df = _derive_neighbor_id_columns(parkings_df, required_neighbor_id_cols)

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
