import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow  # needed for parquet reading
import fastparquet  # needed for parquet reading


def mark_events_vectorized(
        timestamps: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray
) -> np.ndarray:
    """Vectorized check whether timestamps fall inside any event (handles overlapping events).

    Args:
        timestamps: Array of timestamps to check.
        starts: Sorted numpy array of events start timestamps.
        ends: Sorted numpy array of events end timestamps.

    Returns:
        Boolean numpy array indicating active event.
    """
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")

    if len(starts) == 0:
        return np.zeros(len(timestamps), dtype=bool)

    sort_idx = np.argsort(starts)
    starts = starts[sort_idx]
    ends = ends[sort_idx]

    merged_starts = [starts[0]]
    merged_ends = [ends[0]]

    for i in range(1, len(starts)):
        current_start = starts[i]
        current_end = ends[i]

        last_merged_end = merged_ends[-1]

        if current_start <= last_merged_end:

            merged_ends[-1] = max(last_merged_end, current_end)
        else:
            merged_starts.append(current_start)
            merged_ends.append(current_end)

    merged_starts = np.array(merged_starts)
    merged_ends = np.array(merged_ends)


    idx = np.searchsorted(merged_starts, timestamps, side="right") - 1

    valid = idx >= 0
    result = np.zeros(len(timestamps), dtype=bool)

    result[valid] = timestamps[valid] < merged_ends[idx[valid]]

    return result



def prepare_dataframes(
    events_df: pd.DataFrame,
    park_availability_raw: pd.DataFrame,
    sks_raw: pd.DataFrame,
    park_info_df: pd.DataFrame

)-> tuple[pd.DataFrame, pd.DataFrame]:

    #Events dataframe preparation

    min_date_park = pd.to_datetime(park_availability_raw["measured_at"]).min()
    min_date_sks = pd.to_datetime(sks_raw["external_timestamp"]).min()

    cut_off_date = min(min_date_park, min_date_sks)

    events_df["end"] = pd.to_datetime(events_df["end"])

    #Cut events that ended before our parking data (we don't want to waste resources)
    events_df = events_df.loc[events_df["end"] >= cut_off_date].copy()


    #Sort by start timestamp
    events_df = events_df.sort_values("start")


    #Makes an array of start and end dates to compare to SKS and parkings data
    starts = events_df["start"].to_numpy(dtype="datetime64[ns]")
    ends = events_df["end"].to_numpy(dtype="datetime64[ns]")



    #Parkings dataframe preparation


    #Make sure its datetime
    park_availability_raw["measured_at"] = pd.to_datetime(park_availability_raw["measured_at"])

    # Add events flags
    park_availability_raw["is_event"] = mark_events_vectorized(
        park_availability_raw["measured_at"].to_numpy(),
        starts,
        ends,
    )


    park_availability_raw = park_availability_raw.drop_duplicates(subset=["measured_at", "parking_id"])

    # Set index as timestamp for rolling window
    park_availability_raw = park_availability_raw.set_index("measured_at").sort_index()


    # Calculate spaces_occupied instead of spaces left
    capacity_map = park_info_df.set_index("id")["places"]
    park_availability_raw["spaces_occupied"] = park_availability_raw["parking_id"].map(capacity_map) - park_availability_raw["spaces_left"]
    park_availability_raw["spaces_occupied"] = park_availability_raw["spaces_occupied"].clip(lower=0)



    #Prepare SKS df


    #Make sure its timestamp
    sks_raw["external_timestamp"] = pd.to_datetime(sks_raw["external_timestamp"])

    # Add events flags
    sks_raw["is_event"] = mark_events_vectorized(
        sks_raw["external_timestamp"].to_numpy(),
        starts,
        ends,
    )


    sks_raw = sks_raw.drop_duplicates(subset=["external_timestamp"])
    sks_raw = sks_raw.set_index("external_timestamp").sort_index()


    return sks_raw, park_availability_raw


class IQROutlierDetector:
    """Detects outliers using a rolling Interquartile Range (IQR)."""

    def __init__(
            self,
            value_column: str,
            group_column: str | None = None,
            window: str = "7D",
            multiplier: float = 2.0,
            copy: bool = True
    ):
        """
        Args:
            value_column: The name of the column to analyze for outliers.
            group_column: Optional column name for grouping data before analysis.
            window: The size of the rolling window (e.g., '7D' for 7 days).
            multiplier: The IQR multiplier to control outlier sensitivity.
            copy: Whether to operate on a copy of the dataframe. If False, modifies in-place.


            Usage:
            On 1st run do fit(all_data).
            For training and live usage use transform(dataframe with timestamp),
            it will automatically calculate only the last window without data leakage.
        """




        self.value_column = value_column
        self.group_column = group_column
        self.window = window
        self.multiplier = multiplier
        self.copy = copy
        self.train_history_ = None

    def fit(self, df: pd.DataFrame):
        """
        Saves the tail of the training data to prevent cold-start on transform.

        Args:
            df: The training dataframe. Must have a DatetimeIndex.

        Returns:
            self: The fitted detector instance.
        """
        if not df.empty:
            cutoff_time = df.index.max() - pd.Timedelta(self.window)
            self.train_history_ = df[df.index >= cutoff_time].copy()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the rolling IQR logic to detect outliers in the dataset.

        Args:
            df: The dataframe to transform. Must have a DatetimeIndex.

        Returns:
            pd.DataFrame: A dataframe with outlier flags added.
        """
        # Decide whether to modify the original dataframe or a copy
        result_df = df.copy() if self.copy else df

        if result_df.empty:
            return result_df

        original_length = len(result_df)

        # Prepend training history to provide context for the initial rolling windows
        if self.train_history_ is not None and result_df.index.min() > self.train_history_.index.min():
            calc_df = pd.concat([self.train_history_, result_df]).sort_index()
        else:
            calc_df = result_df

        # Extract only the values we need and clip them to 0 (avoids mutating original df)
        clipped_values = calc_df[self.value_column].clip(lower=0)

        if self.group_column is None:
            #Short path when we don't have parking_id

            rolling_window = clipped_values.rolling(self.window, min_periods=1, closed='left')
            q1 = rolling_window.quantile(0.25)
            q3 = rolling_window.quantile(0.75)
        else:
            #Longer path when we do have different parkings
            rolling_grouped = clipped_values.groupby(calc_df[self.group_column])
            q1 = rolling_grouped.transform(
                lambda x: x.rolling(self.window, min_periods=1, closed='left').quantile(0.25))
            q3 = rolling_grouped.transform(
                lambda x: x.rolling(self.window, min_periods=1, closed='left').quantile(0.75))

        iqr = q3 - q1
        lower_bound = (q1 - self.multiplier * iqr).clip(lower=0)
        upper_bound = q3 + self.multiplier * iqr

        # .iloc[-original_length:] ensures we only append data matching the input df size
        result_df["iqr_lower"] = lower_bound.iloc[-original_length:]
        result_df["iqr_upper"] = upper_bound.iloc[-original_length:]

        is_too_low = result_df[self.value_column] < result_df["iqr_lower"]
        is_too_high = result_df[self.value_column] > result_df["iqr_upper"]
        result_df["is_outlier_iqr"] = is_too_low | is_too_high

        if "is_event" in result_df.columns:
            result_df["is_event_iqr_outlier"] = result_df["is_outlier_iqr"] & result_df["is_event"]

        return result_df




#Example usage
def example():
    
    sks_df, park_df = prepare_dataframes(events_raw, park_availability_raw,sks_raw, park_info)
    detector = IQROutlierDetector(
        value_column="spaces_occupied",
        group_column="parking_id",
        window="7D",
        multiplier=2.0
    )

    train_df = detector.fit(park_df).transform(park_df)
    test_df=detector.transform(test_df)



