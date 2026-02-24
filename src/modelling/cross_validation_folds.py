from typing import Dict, List, Optional, Union

import pandas as pd

# Supported frequency units mapped to human-readable labels (for error messages)
FREQ_LABELS = {
    "D": "days",
    "h": "hours",
    "min": "minutes",
}


def format_timestamp(ts: pd.Timestamp, freq: str) -> str:
    """Return a string representation appropriate for the chosen frequency."""
    if freq == "D":
        return ts.strftime("%Y-%m-%d")
    
    # Sub-day: include time; append UTC offset when timezone-aware
    fmt = "%Y-%m-%d %H:%M:%S"
    if ts.tz is not None:
        fmt += "%z"
    return ts.strftime(fmt)


def generate_ts_folds(
    cutoff_end: Union[str, pd.Timestamp],
    train_start_date: Union[str, pd.Timestamp],
    n_folds: int = 3,
    val_periods: int = 28,
    step_periods: Optional[int] = None,
    gap_periods: int = 0,
    min_train_periods: int = 365,
    lookback_periods: int = 0,
    freq: str = "D",
    most_recent_first: bool = True,
) -> List[Dict[str, str]]:
    """
    Generates time series cross-validation folds using an expanding window strategy.

    Supports day-level as well as sub-day granularity (hours, minutes).

    Args:
        cutoff_end (str or Timestamp): The latest timestamp available in the dataset.
        train_start_date (str or Timestamp): The earliest timestamp to start training from (anchor).
        n_folds (int): Number of validation folds to generate. Default 3.
        val_periods (int): Length of the validation window expressed in ``freq`` units. Default 28.
        step_periods (int): Step size between folds in ``freq`` units. Defaults to ``val_periods``.
        gap_periods (int): Gap between training and validation in ``freq`` units. Default 0.
        min_train_periods (int): Minimum required training length in ``freq`` units. Default 365.
        lookback_periods (int): Number of ``freq`` units to shift ``train_start_date`` forward
            so that lag / rolling-window features are valid from the first training row.
            Typically the output of ``required_lookback_periods()``. Default 0.
        freq (str): Time unit for all period parameters. One of ``'D'`` (days),
            ``'h'`` (hours), ``'min'`` (minutes). Default ``'D'``.
        most_recent_first (bool): If True, returns folds starting from the latest date.

    Returns:
        list[dict]: Each dict has keys ``'train_start'``, ``'train_end'``,
            ``'val_start'``, ``'val_end'`` as formatted strings.

    Raises:
        ValueError: If parameters result in invalid date ranges or no folds can be created.
    """

    # Validate parameters
    if freq not in FREQ_LABELS:
        raise ValueError(f"freq must be one of {list(FREQ_LABELS.keys())}, got '{freq}'.")
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1.")
    if val_periods < 1:
        raise ValueError("val_periods must be >= 1.")
    if gap_periods < 0:
        raise ValueError("gap_periods must be >= 0.")
    if min_train_periods < 1:
        raise ValueError("min_train_periods must be >= 1.")
    if lookback_periods < 0:
        raise ValueError("lookback_periods must be >= 0.")

    unit_label = FREQ_LABELS[freq]

    # Standardize inputs to pandas timestamps
    cutoff_end = pd.to_datetime(cutoff_end)
    train_start_date = pd.to_datetime(train_start_date)

    if (cutoff_end.tz is None) != (train_start_date.tz is None):
        raise ValueError("cutoff_end and train_start_date must both be timezone-aware or both be naive.")
    if cutoff_end.tz is not None and cutoff_end.tz != train_start_date.tz:
        train_start_date = train_start_date.tz_convert(cutoff_end.tz)

    # Derive step_periods default
    if step_periods is None:
        step_periods = val_periods
    if step_periods < 1:
        raise ValueError("step_periods must be >= 1.")

    # Shift train_start forward by the lookback buffer so that all
    # lag / rolling-window features are valid from the first training row.
    if lookback_periods > 0:
        train_start_date = train_start_date + ( lookback_periods * pd.Timedelta(1, unit=freq) )

    if cutoff_end < train_start_date:
        raise ValueError("cutoff_end must be on or after train_start_date (after lookback shift).")

    # Build timedeltas using the chosen frequency
    one_unit = pd.Timedelta(1, unit=freq)

    # Upfront check: enough data for at least one fold?
    required_minimum = lookback_periods + min_train_periods + val_periods + gap_periods
    available = int((cutoff_end - train_start_date) / one_unit) + 1
    if required_minimum > available:
        raise ValueError(
            f"Not enough data: need at least {required_minimum} {unit_label} "
            f"(lookback={lookback_periods} + min_train={min_train_periods} + "
            f"gap={gap_periods} + val={val_periods}), but only {available} {unit_label} available."
        )

    folds = []

    # Loop to generate each fold starting from the most recent data
    for k in range(n_folds):
        # Calculate validation boundaries
        # Fold 0 ends at cutoff_end, Fold 1 ends (step_periods) earlier, etc.
        val_end = cutoff_end - step_periods * k * one_unit
        val_start = val_end - (val_periods - 1) * one_unit

        # Calculate training boundaries (ends one unit before the gap starts)
        train_end = val_start - (gap_periods + 1) * one_unit

        # Check if training ends before it even starts
        if train_end < train_start_date:
            raise ValueError(
                f"Cannot generate requested folds: fold {k} training end ({train_end}) is before "
                f"train_start_date ({train_start_date}). Reduce n_folds, val_periods, gap_periods, "
                "or move train_start_date earlier."
            )

        # Check if training period meets the minimum length requirement
        train_length = int((train_end - train_start_date) / one_unit) + 1
        if train_length < min_train_periods:
            raise ValueError(
                f"Cannot generate requested folds: fold {k} training length ({train_length} {unit_label}) "
                f"is shorter than min_train_periods ({min_train_periods}). Reduce n_folds, val_periods, "
                "gap_periods, or min_train_periods, or move train_start_date earlier."
            )

        # Append the valid fold to the list
        folds.append({
            "train_start": format_timestamp(train_start_date, freq),
            "train_end": format_timestamp(train_end, freq),
            "val_start": format_timestamp(val_start, freq),
            "val_end": format_timestamp(val_end, freq),
        })

    # Final check if any folds were generated
    if not folds:
        raise ValueError(
            "Critical Error: No folds could be generated with the provided parameters. "
            "Check if your date range is wide enough for the requested n_folds and min_train_periods."
        )

    return folds if most_recent_first else folds[::-1]