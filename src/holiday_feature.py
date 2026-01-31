import pandas as pd
import numpy as np
import holidays

def add_holiday_features(df: pd.DataFrame, date_col: str = 'measured_at',copy: bool = False) -> pd.DataFrame:
    """
    Holiday feature engineering for parking forecasting.
    
    Arguments:
    - df: Input DataFrame
    - date_col: name of the time column (default 'measured_at')
    
    Returns:
    - DataFrame with new holiday-based features
    
    Features created:
    - is_holiday: Binary flag if the day is a holiday
    - days_since_holiday: Days elapsed since last holiday
    - days_until_holiday: Days until next holiday
    - is_long_weekend_bridge: Monday if Friday or Tuesday is a holiday, Friday if Thursday or Monday is a holiday
    """
    
    if copy:
        df = df.copy()
    
    # Convert date column to datetime if not already in this format
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            raise ValueError(f"Failed to convert '{date_col}' to datetime: {str(e)}")

    # Sort by date 
    df = df.sort_values(by=date_col)
    
    # Normalize date to datetime64[ns] for consistent operations
    df['date_normalized'] = df[date_col].dt.normalize()
    
    # Remove timezone from date_normalized if present
    if df['date_normalized'].dt.tz is not None:
        df['date_normalized'] = df['date_normalized'].dt.tz_localize(None)
    
    # Generate holidays for the data range
    years = df[date_col].dt.year.unique()
    country_holidays = holidays.country_holidays('PL', years=years)
    
    # Extract holiday dates
    holiday_dates = sorted(list(country_holidays.keys()))
    
    # Create holiday dataframe early for use in feature creation
    holiday_df = pd.DataFrame({'holiday_date': pd.to_datetime(holiday_dates)})
    holiday_df['holiday_date'] = holiday_df['holiday_date'].dt.normalize()
    
    
    # Creating Features
    
    # 1. Holiday flag 
    df['is_holiday'] = df['date_normalized'].isin(holiday_df['holiday_date']).astype(int)
    
    # 2. Days to/from holidays
    
    # Convert to numeric values for efficient searching
    holiday_dates_numeric = holiday_df['holiday_date'].values.astype('datetime64[D]').astype(int)
    dates_numeric = df['date_normalized'].values.astype('datetime64[D]').astype(int)
    
    # Find insertion positions using searchsorted
    positions = np.searchsorted(holiday_dates_numeric, dates_numeric)
    
    # Calculate days since and until holidays
    prev_positions = np.maximum(positions - 1, 0)
    next_positions = np.minimum(positions, len(holiday_dates_numeric) - 1)
    
    prev_holiday_dates = holiday_df['holiday_date'].values[prev_positions]
    next_holiday_dates = holiday_df['holiday_date'].values[next_positions]
    
    # Handle edge cases where there's no previous/next holiday
    df['days_since_holiday'] = np.where(
        positions > 0,
        (dates_numeric - prev_holiday_dates.astype('datetime64[D]').astype(int)),
        999 # Fill value if no previous holiday
    ).astype(int)
    
    df['days_until_holiday'] = np.where(
        positions < len(holiday_dates_numeric),
        (next_holiday_dates.astype('datetime64[D]').astype(int) - dates_numeric),
        999 # Fill value if no next holiday
    ).astype(int)
    
    # Long Weekend Bridges Feature
    # Monday is a bridge day if Friday (3 days before) or Tuesday (1 day after) is a holiday
    # Friday is a bridge day if Thursday (1 day before) or Monday (3 days after) is a holiday
    day_of_week = df[date_col].dt.dayofweek  # 0=Monday, 4=Friday, 6=Sunday
    
    bridge_monday = (day_of_week == 0) & (
        (df['days_since_holiday'] == 3) |  # Friday was a holiday
        (df['days_until_holiday'] == 1)    # Tuesday is a holiday
    )
    
    bridge_friday = (day_of_week == 4) & (
        (df['days_since_holiday'] == 1) |  # Thursday was a holiday
        (df['days_until_holiday'] == 3)    # Monday is a holiday
    )
    
    df['is_long_weekend_bridge'] = (bridge_monday | bridge_friday).astype(int) 

    # Clean up useless columns
    cols_to_drop = ['date_normalized']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    return df