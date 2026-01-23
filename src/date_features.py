import pandas as pd
import numpy as np

def add_date_features(df: pd.DataFrame, date_col: str = 'measured_at') -> pd.DataFrame:
    """
    Creates features based on date.
    Arguments:
    - df: Input DataFrame
    - date_col: name of the time column (default 'measured_at', as in the original dataset)
    
    Returns:
    - DataFrame with new columns
    """
    
    df = df.copy()
    
    # Convert date column to datetime if not already in this format
    if df[date_col].dtype == 'object':
        df[date_col] = pd.to_datetime(df[date_col])

    # Standard date features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['day_in_the_month'] = df[date_col].dt.day # 0=Monday, 6=Sunday
    df['week_of_the_year'] = df[date_col].dt.isocalendar().week.astype(int)
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Features with finer granularity
    df['hour'] = df[date_col].dt.hour
    df['minute'] = df[date_col].dt.minute

    # Binary features with values 0/1
    df['is_working_hour'] = df['hour'].between(7, 15).astype(int)
    df['is_rush_hour'] = (df['hour'].between(6, 9)).astype(int)

    return df