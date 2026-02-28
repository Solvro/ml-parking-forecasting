import pandas as pd

from src.feature_engineering.date_features import add_date_features
from src.feature_engineering.holiday_feature import add_holiday_features


def build_features_for_range(
    base_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    lags=(7, 14, 28, 365),
    roll_lags=(7, 14, 28),
    roll_windows=(7, 14),
    roll_stats=("mean", "std"),
    copy: bool = False,
) -> pd.DataFrame:
    """Build a complete feature DataFrame for a specified date range.

    Orchestrates all feature engineering steps: calendar/time features and
    Polish holiday features, 
    TODO - lag and rolling-window features will be added
    later by other contributors.

    Processing pipeline:
        1. Validate inputs.
        2. Slice base_df to [start_date, end_date].
        3. Add date/time features via add_date_features().
        4. Add Polish holiday features via add_holiday_features().
        5. (placeholder) Lag features — TODO.
        6. (placeholder) Rolling-window features — TODO.
        7. (placeholder) Additional features — TODO.
        8. Return the enriched DataFrame.

    Args:
        base_df (pd.DataFrame): Source data with at least a 'date_col' column
            (datetime) and a 'target_col' column (target value).
            May contain additional columns that will be preserved.
        start_date (str): First date of the output range,
            in 'YYYY-MM-DD' format.
        end_date (str): Last date of the output range,
            in 'YYYY-MM-DD' format.
        lags (tuple[int, ...]): Simple lag offsets applied to 'target_col'.
            Default (7, 14, 28, 365). — TODO: not yet used.
        roll_lags (tuple[int, ...]): Starting offsets for rolling-window
            features. Default (7, 14, 28). — TODO: not yet used.
        roll_windows (tuple[int, ...]): Window sizes for rolling statistics.
            Default (7, 14). — TODO: not yet used.
        roll_stats (tuple[str, ...]): Aggregation functions for rolling
            windows. Default ("mean", "std"). — TODO: not yet used.
        copy (bool): If True, operate on a copy of base_df to avoid
            mutating the original. Default False (modifies in place for
            better memory efficiency on large frames).

    Returns:
        pd.DataFrame: Feature DataFrame limited to [start_date, end_date].
            Contains the original columns plus all generated features.

    Raises:
        ValueError: If date range is invalid.
        KeyError: If base_df does not contain 'date_col' or 'target_col' columns.
    """

    # 1. Validate inputs
    if "date_col" not in base_df.columns or "target_col" not in base_df.columns:
        raise KeyError("base_df must contain 'date_col' and 'target_col' columns.")

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if end < start:
        raise ValueError(
            f"end_date ({end_date}) must be on or after start_date ({start_date})."
        )
    
    # 2. Slice to the target range
    df = base_df.copy() if copy else base_df
    df["date_col"] = pd.to_datetime(df["date_col"])
    df = df.sort_values("date_col").reset_index(drop=True)

    mask = (df["date_col"] >= start) & (df["date_col"] <= end)
    df = df.loc[mask].reset_index(drop=True)

    # 3. Feature engineering functions:
    
    df = add_date_features(df, date_col="date_col") # adds calendar and time features based on the 'date_col' column

    df = add_holiday_features(df, date_col="date_col") # adds features indicating whether the date is a Polish holiday based on the 'date_col' column

    # TODO: Other features developed by other contributors (lags, rolling windows, etc.) would be added here.
    

    return df
