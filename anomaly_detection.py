import pandas as pd
import numpy as np
import pyarrow  # needed for parquet reading
import fastparquet  # needed for parquet reading


def prepare_events_df(
        df: pd.DataFrame,
        cut_off_date: pd.Timestamp,
        copy: bool = True,
) -> pd.DataFrame:
    """Filter events to only those relevant for the analysis window.

    Args:
        df: Raw events dataframe.
        cut_off_date: Minimal timestamp present in other datasets.
        copy: Whether to operate on a copy of the dataframe.

    Returns:
        Filtered events dataframe.
    """
    if copy:
        df = df.copy()

    # Normalize dtypes to make sure we don't get any issues later
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])

    # Cut events so later we don't have to check events that are before our parking data
    return df[df["end"] >= cut_off_date]


def build_event_index(events_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build sorted numpy arrays of event boundaries.


    Args:
        events_df: Filtered events dataframe.

    Returns:
        Tuple of (starts, ends) numpy arrays sorted by start time.
    """

    # Empty guard
    if events_df.empty:
        return (
            np.array([], dtype="datetime64[ns]"),
            np.array([], dtype="datetime64[ns]"),
        )

    # We must sort the values first
    events_df = events_df.sort_values("start")

    starts = events_df["start"].to_numpy(dtype="datetime64[ns]")
    ends = events_df["end"].to_numpy(dtype="datetime64[ns]")

    return starts, ends


def mark_events_vectorized(
        timestamps: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
) -> np.ndarray:
    """Vectorized check whether timestamps fall inside any event.

    Args:
        timestamps: Array of timestamps to check.
        starts: Sorted array of event start times.
        ends: Sorted array of event end times.

    Returns:
        Boolean numpy array indicating active event.
    """

    # make sure everything is a datetime
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")

    # Check if is empty
    if len(starts) == 0:
        return np.zeros(len(timestamps), dtype=bool)

    # Searches the last event before the timestamp
    idx = np.searchsorted(starts, timestamps, side="right") - 1

    valid = idx >= 0
    result = np.zeros(len(timestamps), dtype=bool)

    # If event is in timestamp then its valid
    result[valid] = timestamps[valid] < ends[idx[valid]]

    return result


def prepare_park_availability_df(
        df: pd.DataFrame,
        park_info_df: pd.DataFrame,
        starts: np.ndarray,
        ends: np.ndarray,
        copy: bool = True,
) -> pd.DataFrame:
    """Prepare parking availability dataset with event flags.



    Args:
        df: Raw parking availability dataframe.
        park_info_df: Parking metadata.
        starts: Event start times.
        ends: Event end times.
        copy: Whether to operate on a copy of the dataframe.

    Returns:
        Enriched parking dataframe.
    """
    if copy:
        df = df.copy()

    # Normalize and make sure everything is datetime
    df["measured_at"] = pd.to_datetime(df["measured_at"])

    # Add events flags
    df["is_event"] = mark_events_vectorized(
        df["measured_at"].to_numpy(),
        starts,
        ends,
    )

    # Set index as timestamp for rolling window
    df = df.set_index("measured_at").sort_index()

    # Calculate spaces_occupied instead of spaces left
    capacity_map = park_info_df.set_index("id")["places"]
    df["spaces_occupied"] = df["parking_id"].map(capacity_map) - df["spaces_left"]

    return df


def prepare_sks_df(
        df: pd.DataFrame,
        starts: np.ndarray,
        ends: np.ndarray,
        copy: bool = True,
) -> pd.DataFrame:
    """Prepare SKS dataset with event flags.


    Args:
        df: Raw SKS dataframe.
        starts: Event start times.
        ends: Event end times.
        copy: Whether to operate on a copy of the dataframe.

    Returns:
        Enriched SKS dataframe.
    """
    if copy:
        df = df.copy()
    # Normalize and make sure everything is datetime

    df["external_timestamp"] = pd.to_datetime(df["external_timestamp"])

    # Add events flags

    df["is_event"] = mark_events_vectorized(
        df["external_timestamp"].to_numpy(),
        starts,
        ends,
    )

    df = df.set_index("external_timestamp").sort_index()
    return df


class IQROutlierDetector:
    """Detect outliers using rolling IQR."""

    def __init__(
            self,
            value_column: str,
            filter_column: str | None = None,
            window: str = "7D",
            multiplier: float = 2.0,
            copy: bool = True,
    ):
        """
        Args:
            value_column: Column to analyze.
            filter_column: Optional grouping column.
            window: Rolling window (time-based).
            multiplier: IQR multiplier controlling sensitivity.
            copy: Whether to operate on a copy of the dataframe.
        """
        self.value_column = value_column
        self.filter_column = filter_column
        self.window = window
        self.multiplier = multiplier
        self.copy = copy
        self.train_history_ = None

    def fit(self, df: pd.DataFrame, y=None):
        """Save the tail of the training data to prevent cold-start on transform."""
        if not df.empty:
            last_timestamp = df.index.max()
            cutoff = last_timestamp - pd.Timedelta(self.window)
            self.train_history_ = df[df.index >= cutoff].copy()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns:
            Dataframe with outlier flags added.
        """
        if self.copy:
            df = df.copy()

        # Combine with train history to avoid cold start on rolling calculation
        if self.train_history_ is not None and not df.empty:
            # Only prepend if we are transforming new, non-overlapping data (like a test set)
            if df.index.min() > self.train_history_.index.min():
                calc_df = pd.concat([self.train_history_, df]).sort_index()
            else:
                calc_df = df
        else:
            calc_df = df

        # there can't be below 0 people in a buidling or parking
        calc_df[self.value_column] = calc_df[self.value_column].clip(lower=0)

        # Fast path when no filtering is required
        if self.filter_column is None:
            rolling_obj = calc_df[self.value_column].rolling(self.window, min_periods=1)
            q1 = rolling_obj.quantile(0.25)
            q3 = rolling_obj.quantile(0.75)
        else:
            # More expensive path that requires filtering (for eg. per parking)
            q1 = (
                calc_df.groupby(self.filter_column)[self.value_column]
                .rolling(self.window, min_periods=1)
                .quantile(0.25)
                .reset_index(level=0, drop=True)
            )

            q3 = (
                calc_df.groupby(self.filter_column)[self.value_column]
                .rolling(self.window, min_periods=1)
                .quantile(0.75)
                .reset_index(level=0, drop=True)
            )

        # Calculate IQR
        iqr = q3 - q1
        # Calculate bounds using multiplier
        lower_bound = (q1 - self.multiplier * iqr).clip(lower=0)
        upper_bound = q3 + self.multiplier * iqr

        # Slice the bounds back to the original df's length to ignore the prepended train_history
        df["iqr_lower"] = lower_bound.iloc[-len(df):]
        df["iqr_upper"] = upper_bound.iloc[-len(df):]

        df["is_outlier_iqr"] = (
                (df[self.value_column] < df["iqr_lower"])
                | (df[self.value_column] > df["iqr_upper"])
        )
        # Check if both there is event and an outlier
        df["is_event_iqr_outlier"] = df["is_outlier_iqr"] & df["is_event"]

        return df