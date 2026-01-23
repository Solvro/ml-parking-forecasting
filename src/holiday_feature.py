import pandas as pd
import numpy as np
import holidays

def add_holiday_features(df: pd.DataFrame, date_col: str = 'measured_at') -> pd.DataFrame:
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
    
    - is_holiday_tomorrow: Will tomorrow be a holiday
    - was_holiday_yesterday: Was yesterday a holiday
    
    - in_holiday_window_7d: Within 7 days of a holiday
    - in_holiday_window_3d: Within 3 days of a holiday
    
    - is_long_weekend_bridge: Monday before Tuesday holiday or Friday after Thursday holiday
    
    - pre_holiday_period: 1-2 days before holiday (last minute rush)
    - post_holiday_period: 1-2 days after holiday (lazy return to work)
    """
    
    df = df.copy()
    
    if df[date_col].dtype == 'object':
        df[date_col] = pd.to_datetime(df[date_col])

    # Sort by date (required for merge_asof)
    df = df.sort_values(by=date_col)
    
    # Generate holidays for the data range
    years = df[date_col].dt.year.unique()
    country_holidays = holidays.country_holidays('PL', years=years)
    
    # Extract date-only for holiday matching
    df['date_only'] = df[date_col].dt.date
    holiday_dates = sorted(list(country_holidays.keys()))
    
    
    # Creating Features
    
    # 1. Holiday flag
    df['is_holiday'] = df['date_only'].apply(lambda x: 1 if x in country_holidays else 0).astype(int)
    
    # 2. Days to/from holidays
    holiday_df = pd.DataFrame({'holiday_date': pd.to_datetime(holiday_dates)})
    holiday_df['holiday_date'] = holiday_df['holiday_date'].dt.normalize()
    df['date_normalized'] = df[date_col].dt.normalize()
    
    # Remove timezone from both dataframes for merge compatibility
    if df['date_normalized'].dt.tz is not None:
        df['date_normalized'] = df['date_normalized'].dt.tz_localize(None)
    if holiday_df['holiday_date'].dt.tz is not None:
        holiday_df['holiday_date'] = holiday_df['holiday_date'].dt.tz_localize(None)
    
    # Days since last holiday
    df = pd.merge_asof(
        df,
        holiday_df.rename(columns={'holiday_date': 'prev_holiday'}),
        left_on='date_normalized',
        right_on='prev_holiday',
        direction='backward'
    )
    df['days_since_holiday'] = (df['date_normalized'] - df['prev_holiday']).dt.days
    
    # Days until next holiday
    df = pd.merge_asof(
        df,
        holiday_df.rename(columns={'holiday_date': 'next_holiday'}),
        left_on='date_normalized',
        right_on='next_holiday',
        direction='forward'
    )
    df['days_until_holiday'] = (df['next_holiday'] - df['date_normalized']).dt.days
    
    # 3. Tomorrow/Yesterday holiday flags
    df['is_holiday_tomorrow'] = (df['days_until_holiday'] == 1).astype(int)
    df['was_holiday_yesterday'] = (df['days_since_holiday'] == 1).astype(int)
    
    # 4. Within X days of holiday
    df['in_holiday_window_3d'] = ((df['days_until_holiday'] <= 3) | (df['days_since_holiday'] <= 3)).astype(int)
    
    df['in_holiday_window_7d'] = ((df['days_until_holiday'] <= 7) | (df['days_since_holiday'] <= 7)).astype(int)
    
    # 5. Behavioral changes around holidays
    df['pre_holiday_period'] = ((df['days_until_holiday'] >= 1) & (df['days_until_holiday'] <= 2)).astype(int)  # 1-2 days before

    df['post_holiday_period'] = ((df['days_since_holiday'] >= 1) & (df['days_since_holiday'] <= 2)).astype(int)  # 1-2 days after

    # Long Weekend Bridges
    day_of_week = df[date_col].dt.dayofweek  # 0=Mon, 6=Sun
    
    # Monday is free if Tuesday is holiday and Friday is free if Thursday was holiday
    df['is_long_weekend_bridge'] = (((day_of_week == 0) & (df['is_holiday_tomorrow'] == 1)) | ((day_of_week == 4) & (df['was_holiday_yesterday'] == 1))).astype(int) 
    

    # Clean up useless columns
    cols_to_drop = ['date_only', 'prev_holiday', 'next_holiday', 'date_normalized']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    return df