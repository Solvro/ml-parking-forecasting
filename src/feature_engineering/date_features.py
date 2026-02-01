import pandas as pd
import numpy as np

def add_date_features(df: pd.DataFrame, date_col: str = 'measured_at', copy: bool = False, 
                      work_start: int = 7, work_end: int = 15, rush_hours: tuple = (6, 9)) -> pd.DataFrame:
    """
    Creates features based on date.
    Arguments:
    - df: Input DataFrame
    - date_col: name of the time column (default 'measured_at', as in the original dataset)
    - copy: whether to copy the DataFrame before modifying (default False)
    - work_start: start hour of working hours (default 7)
    - work_end: end hour of working hours (default 15)
    - rush_hours: tuple of (start, end) hours for rush hour (default (6, 9))
    
    Returns:
    - DataFrame with new columns
    """
    
    if copy:
        df = df.copy()
    
    # Convert date column to datetime if not already in this format
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            raise ValueError(f"Failed to convert '{date_col}' to datetime: {str(e)}")

    # Standard date features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['day_in_the_month'] = df[date_col].dt.day 
    df['week_of_the_year'] = df[date_col].dt.isocalendar().week.astype(int)
    df['day_of_week'] = df[date_col].dt.dayofweek # 0=Monday, 6=Sunday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Features with finer granularity
    df['hour'] = df[date_col].dt.hour
    # Only extract minute if data has minute-level granularity
    if df[date_col].dt.minute.nunique() > 1:
        df['minute'] = df[date_col].dt.minute

    # Binary features with values 0/1
    df['is_working_hour'] = df['hour'].between(work_start, work_end).astype(int)
    df['is_rush_hour'] = (df['hour'].between(rush_hours[0], rush_hours[1])).astype(int)

    return df