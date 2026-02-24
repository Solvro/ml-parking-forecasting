from typing import Tuple

def required_lookback_periods(
    lags: Tuple[int, ...],
    roll_lags: Tuple[int, ...],
    roll_windows: Tuple[int, ...],
) -> int:
    
    """Calculate the minimum number of lookback periods required to compute all features.

    Determines the buffer of historical data needed before the first prediction
    date, based on the lag and rolling-window feature definitions.

    Args:
        lags (tuple[int, ...]): Simple lag offsets (e.g. 7, 14, 28, 365).
        roll_lags (tuple[int, ...]): Starting offsets for rolling-window features.
        roll_windows (tuple[int, ...]): Window sizes applied at each roll lag.

    Returns:
        int: The number of past time units needed as a warm-up buffer.

    Raises:
        ValueError: If any value is non-positive or roll_lags/roll_windows
            are provided inconsistently (one empty, the other not).
    """

    # Validate that all values are positive integers
    if lags and any(l <= 0 for l in lags):
        raise ValueError("All lag values must be positive integers.")
    if roll_lags and any(rl <= 0 for rl in roll_lags):
        raise ValueError("All roll_lag values must be positive integers.")
    if roll_windows and any(w <= 0 for w in roll_windows):
        raise ValueError("All roll_window values must be positive integers.")

    # Detect inconsistent rolling configuration
    if bool(roll_lags) != bool(roll_windows):
        raise ValueError(
            "roll_lags and roll_windows must both be provided or both be empty. "
            f"Got roll_lags={roll_lags}, roll_windows={roll_windows}."
        )

    max_simple_lag = max(lags) if lags else 0
    max_rolling_lag = max(
        (rl + w - 1 for rl in roll_lags for w in roll_windows),
        default=0,
    )
    return max(max_simple_lag, max_rolling_lag)