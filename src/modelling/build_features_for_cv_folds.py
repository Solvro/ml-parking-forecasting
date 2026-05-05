from typing import Any, Dict, List

import pandas as pd

from src.modelling.build_features_for_range import build_features_for_range
from src.modelling.cross_validation_folds import generate_ts_folds
from src.modelling.lookback_days import required_lookback_periods


def build_features_for_cv_folds(
    base_df: pd.DataFrame,
    df_sks_users: pd.DataFrame,
    df_parkings: pd.DataFrame,
    df_calendar: pd.DataFrame,
    train_start_date: str,
    cutoff_end: str,
    n_folds: int = 3,
    val_periods: int = 28,
    step_periods: int | None = None,
    gap_periods: int = 0,
    min_train_periods: int = 365,
    freq: str = "D",
    most_recent_first: bool = True,
    lags: tuple = (7, 14, 28, 365),
    roll_lags: tuple = (7, 14, 28),
    roll_windows: tuple = (7, 14),
    copy: bool = False,
) -> List[Dict[str, Any]]:
    """Generate feature sets for time-series CV folds using the shared pipeline.

    Workflow:
        1. Compute lookback from lag/rolling definitions.
        2. Generate folds via generate_ts_folds.
        3. For each fold, run build_features_for_range on train range (fitting the pipeline).
        4. Pass the fitted pipeline to build_features_for_range for the val range 
           to prevent data leakage (using transform only).

    Returns:
        list[dict]: Per-fold dictionaries with date boundaries and prepared
            train_df / val_df feature DataFrames.
    """

    lookback_periods = required_lookback_periods(lags, roll_lags, roll_windows)

    folds = generate_ts_folds(
        cutoff_end=cutoff_end,
        train_start_date=train_start_date,
        n_folds=n_folds,
        val_periods=val_periods,
        step_periods=step_periods,
        gap_periods=gap_periods,
        min_train_periods=min_train_periods,
        lookback_periods=lookback_periods,
        freq=freq,
        most_recent_first=most_recent_first,
    )

    results: List[Dict[str, Any]] = []
    
    for fold_idx, fold in enumerate(folds):
        
        # 1. Process Train Data (Learn statistics, return the fitted pipeline)
        train_df, fitted_pipe = build_features_for_range(
            base_df=base_df,
            df_sks_users=df_sks_users,
            df_parkings=df_parkings,
            df_calendar=df_calendar,
            start_date=fold["train_start"],
            end_date=fold["train_end"],
            lags=lags,
            roll_lags=roll_lags,
            roll_windows=roll_windows,
            freq=freq,
            copy=copy,
            return_pipeline=True,
            fitted_pipeline=None,
            apply_lookback=True,
        )

        # 2. Process Validation Data (Apply learned statistics, NO fitting)
        val_df = build_features_for_range(
            base_df=base_df,
            df_sks_users=df_sks_users,
            df_parkings=df_parkings,
            df_calendar=df_calendar,
            start_date=fold["val_start"],
            end_date=fold["val_end"],
            lags=lags,
            roll_lags=roll_lags,
            roll_windows=roll_windows,
            freq=freq,
            copy=copy,
            return_pipeline=False,
            fitted_pipeline=fitted_pipe,
            apply_lookback=True,
        )

        results.append(
            {
                "fold": fold_idx,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "val_start": fold["val_start"],
                "val_end": fold["val_end"],
                "train_df": train_df,
                "val_df": val_df,
            }
        )

    return results