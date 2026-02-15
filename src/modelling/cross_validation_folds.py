from datetime import timedelta
from typing import Dict, List, Optional, Union

import pandas as pd


def generate_ts_folds(
    cutoff_end: Union[str, pd.Timestamp],
    train_start_date: Union[str, pd.Timestamp],
    n_folds: int = 3,
    val_days: int = 28,
    step_days: Optional[int] = None,
    gap_days: int = 0,
    min_train_days: int = 365,
    most_recent_first: bool = True,
) -> List[Dict[str, str]]:
    """
    Generates time series cross-validation folds using an expanding window strategy.
    
    Args:
        cutoff_end (str or datetime): The latest date available in the dataset.
        train_start_date (str or datetime): The earliest date to start training from (anchor).
        n_folds (int): Number of validation folds to generate. Default 3.
        val_days (int): Length of the validation period in days. Default 28.
        step_days (int): Step size between folds. Defaults to val_days if None.
        gap_days (int): Number of days to skip between training and validation. Default 0.
        min_train_days (int): Minimum required length of the training period. Default 365.
        most_recent_first (bool): If True, returns folds starting from the latest date.

    Returns:
        list: A list of dictionaries, each containing 'train_start', 'train_end', 
              'val_start', and 'val_end' as ISO strings.
              
    Raises:
        ValueError: If parameters result in invalid date ranges or no folds can be created.
    """

    if n_folds < 1:
        raise ValueError("n_folds must be >= 1.")
    if val_days < 1:
        raise ValueError("val_days must be >= 1.")
    if gap_days < 0:
        raise ValueError("gap_days must be >= 0.")
    if min_train_days < 1:
        raise ValueError("min_train_days must be >= 1.")

    # Standardize inputs to pandas timestamps
    cutoff_end = pd.to_datetime(cutoff_end)
    train_start_date = pd.to_datetime(train_start_date)

    if (cutoff_end.tz is None) != (train_start_date.tz is None):
        raise ValueError("cutoff_end and train_start_date must both be timezone-aware or both be naive.")
    if cutoff_end.tz is not None and cutoff_end.tz != train_start_date.tz:
        train_start_date = train_start_date.tz_convert(cutoff_end.tz)
    
    # If step_days is not provided, we slide the window by the validation length
    if step_days is None:
        step_days = val_days
    if step_days < 1:
        raise ValueError("step_days must be >= 1.")

    if cutoff_end < train_start_date:
        raise ValueError("cutoff_end must be on or after train_start_date.")
        
    folds = []

    # Loop to generate each fold starting from the most recent data
    for k in range(n_folds):
        # Calculate validation boundaries
        # Fold 0 ends at cutoff_end, Fold 1 ends (step_days) earlier, etc.
        val_end = cutoff_end - timedelta(days=k * step_days)
        val_start = val_end - timedelta(days=val_days - 1)
        
        # Calculate training boundaries (ends before the gap starts)
        train_end = val_start - timedelta(days=gap_days + 1)
        
        # Check if training ends before it even starts
        if train_end < train_start_date:
            raise ValueError(
                f"Cannot generate requested folds: fold {k} training end ({train_end}) is before "
                f"train_start_date ({train_start_date}). Reduce n_folds, val_days, gap_days, "
                "or move train_start_date earlier."
            )
            
        # Check if training period meets the minimum length requirement
        train_length = (train_end - train_start_date).days + 1
        if train_length < min_train_days:
            raise ValueError(
                f"Cannot generate requested folds: fold {k} training length ({train_length} days) "
                f"is shorter than min_train_days ({min_train_days}). Reduce n_folds, val_days, "
                "gap_days, or min_train_days, or move train_start_date earlier."
            )

        # Append the valid fold to the list
        folds.append({
            "train_start": train_start_date.strftime('%Y-%m-%d'),
            "train_end": train_end.strftime('%Y-%m-%d'),
            "val_start": val_start.strftime('%Y-%m-%d'),
            "val_end": val_end.strftime('%Y-%m-%d')
        })

    # Final check if any folds were generated
    if not folds:
        raise ValueError("Critical Error: No folds could be generated with the provided parameters. "
                         "Check if your date range is wide enough for the requested n_folds and min_train_days.")

    return folds if most_recent_first else folds[::-1]