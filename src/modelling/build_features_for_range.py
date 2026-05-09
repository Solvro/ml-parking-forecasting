from typing import Optional, Tuple, Union

import pandas as pd
from sklearn.pipeline import Pipeline

from src.feature_engineering.feature_pipeline import build_feature_pipeline, FeaturePipelineConfig
from src.modelling.lookback_days import required_lookback_periods


def build_features_for_range(
    base_df: pd.DataFrame,
    df_sks_users: pd.DataFrame,
    df_parkings: pd.DataFrame,
    df_calendar: pd.DataFrame,
    start_date: str,
    end_date: str,
    lags: tuple = (7, 14, 28, 365),
    roll_lags: tuple = (7, 14, 28),
    roll_windows: tuple = (7, 14),
    freq: str = "D",
    copy: bool = False,
    fitted_pipeline: Optional[Pipeline] = None,
    return_pipeline: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Pipeline]]:
    """Build features for a single date range using the shared pipeline.

    The function includes a historical lookback buffer before start_date
    so that lag/rolling features are valid from the first requested row.
    It supports stateful ML pipelines to prevent data leakage during CV.

    Args:
        base_df (pd.DataFrame): Source parking observations.
        df_sks_users (pd.DataFrame): SKS users data required by pipeline mergers.
        df_parkings (pd.DataFrame): Parking metadata required by pipeline steps.
        df_calendar (pd.DataFrame): Calendar/events data required by pipeline steps.
        start_date (str): First date of the output range ('YYYY-MM-DD').
        end_date (str): Last date of the output range ('YYYY-MM-DD').
        lags (tuple): Lag offsets used to compute required lookback.
        roll_lags (tuple): Starting offsets for rolling-window features.
        roll_windows (tuple): Window sizes for rolling statistics.
        freq (str): Time unit for lookback computation (e.g., 'D', 'h', 'min').
        copy (bool): If True, operate on a copy of base_df.
        fitted_pipeline (Pipeline, optional): A pre-fitted scikit-learn pipeline.
            If provided, the pipeline will only use .transform() to prevent
            data leakage. If None, a new pipeline is created and .fit_transform()
            is called. Defaults to None.
        return_pipeline (bool): If True, returns a tuple of (DataFrame, Pipeline).
            Defaults to False.

    Returns:
        pd.DataFrame or Tuple[pd.DataFrame, Pipeline]: The enriched DataFrame 
        limited to [start_date, end_date], and optionally the fitted pipeline.

    Raises:
        ValueError: If date range is invalid.
        KeyError: If required columns are missing.
    """

    # 1. Validate inputs
    required_cols = {"measured_at", "parking_id"}
    missing_cols = required_cols.difference(base_df.columns)
    if missing_cols:
        raise KeyError(f"base_df is missing required columns: {sorted(missing_cols)}.")

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if end < start:
        raise ValueError(f"end_date ({end_date}) must be on or after start_date ({start_date}).")

    # 2. Timezone handling (Extract timezone from the base dataset)
    # Using a fast metadata check instead of converting the whole column
    series_tz = base_df["measured_at"].dt.tz if pd.api.types.is_datetime64_any_dtype(base_df["measured_at"]) else pd.to_datetime(base_df["measured_at"].iloc[0]).tz

    if series_tz is not None:
        start = start.tz_localize(series_tz) if start.tzinfo is None else start.tz_convert(series_tz)
        end = end.tz_localize(series_tz) if end.tzinfo is None else end.tz_convert(series_tz)
    else:
        start = start.tz_convert(None) if start.tzinfo is not None else start
        end = end.tz_convert(None) if end.tzinfo is not None else end

    # 3. Compute lookback and slice efficiently
    lookback_periods = required_lookback_periods(lags, roll_lags, roll_windows)
    lookback_start = start - (lookback_periods * pd.Timedelta(1, unit=freq))

    # Optimization: Create mask on datetime series before sorting the whole dataframe
    temp_dates = pd.to_datetime(base_df["measured_at"])
    mask = (temp_dates >= lookback_start) & (temp_dates <= end)
    
    # Safely slice and copy
    df = base_df.loc[mask].copy() if copy else base_df.loc[mask]
    df["measured_at"] = pd.to_datetime(df["measured_at"])
    df = df.sort_values("measured_at").reset_index(drop=True)

    # 4. Run the shared feature pipeline (ML safe)
    config = FeaturePipelineConfig().from_json("../config/default.json")
    if fitted_pipeline is None:
        # Training phase: Instantiate and fit the pipeline
        pipeline = build_feature_pipeline(
            config=config,
            df_sks_users=df_sks_users,
            df_parkings=df_parkings,
            df_calendar=df_calendar,
        )
        df = pipeline.fit_transform(df)
    else:
        # Validation/Test phase: Only transform using the pre-fitted pipeline
        pipeline = fitted_pipeline
        df = pipeline.transform(df)

    # 5. Trim back to the requested output interval (remove lookback buffer)
    df["measured_at"] = pd.to_datetime(df["measured_at"])
    df = df.loc[(df["measured_at"] >= start) & (df["measured_at"] <= end)]
    df = df.reset_index(drop=True)

    if return_pipeline:
        return df, pipeline
    
    return df