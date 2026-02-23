import pandas as pd
import requests
from datetime import datetime
from typing import List, Optional, Tuple

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Wrocław - Plac Grunwaldzki
DEFAULT_LOCATION = (51.1116, 17.0601)

DEFAULT_FEATURES = ["temperature_2m", "rain", "apparent_temperature", "snowfall", "snow_depth", "precipitation", "weather_code", "visibility", "surface_pressure", "wind_speed_10m"]

type Location = Tuple[float, float]


def fetch_weather_forecast(
    location: Location = DEFAULT_LOCATION,
    features: List[str] = DEFAULT_FEATURES
) -> pd.DataFrame:
    """
    Fetch weather forecast data from Open-Meteo API.
    
    Arguments:
    - location: Tuple of (latitude, longitude)
    - features: List of weather features to fetch
    
    Returns:
    - DataFrame with weather forecast data
    """
    response = requests.get(
        f"{FORECAST_URL}?latitude={location[0]}&longitude={location[1]}&hourly={','.join(features)}"
    )
    response.raise_for_status()
    
    df = pd.DataFrame(response.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_historical_weather(
    start_date: str,
    end_date: str,
    location: Location = DEFAULT_LOCATION,
    features: List[str] = DEFAULT_FEATURES
) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo API.
    
    Arguments:
    - start_date: Start date in 'YYYY-MM-DD' format
    - end_date: End date in 'YYYY-MM-DD' format
    - location: Tuple of (latitude, longitude)
    - features: List of weather features to fetch
    
    Returns:
    - DataFrame with historical weather data
    """
    response = requests.get(
        f"{HISTORICAL_URL}?latitude={location[0]}&longitude={location[1]}"
        f"&start_date={start_date}&end_date={end_date}&hourly={','.join(features)}"
    )
    response.raise_for_status()
    
    df = pd.DataFrame(response.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_weather_for_range(
    dates: pd.Series,
    location: Location,
    features: List[str]
) -> pd.DataFrame:
    """
    Internal function to fetch weather data for a date range, handling both
    historical and forecast data automatically.
    
    Arguments:
    - dates: Series of datetime values
    - location: Tuple of (latitude, longitude)
    - features: List of weather features to fetch
    
    Returns:
    - Combined DataFrame with weather data
    """
    today = pd.Timestamp(datetime.now().date())
    min_date = dates.min()
    max_date = dates.max()
    
    weather_dfs = []
    
    # Fetch historical data if needed (dates before today)
    if min_date.date() < today.date():
        hist_end = min(max_date, today - pd.Timedelta(days=1))
        hist_df = fetch_historical_weather(
            min_date.strftime('%Y-%m-%d'),
            hist_end.strftime('%Y-%m-%d'),
            location,
            features
        )
        weather_dfs.append(hist_df)
    
    # Fetch forecast data if needed (dates from today onwards)
    if max_date.date() >= today.date():
        forecast_df = fetch_weather_forecast(location, features)
        weather_dfs.append(forecast_df)
    
    # Combine and deduplicate
    if len(weather_dfs) == 1:
        return weather_dfs[0]
    
    combined = pd.concat(weather_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['time'], keep='last')
    return combined.sort_values('time').reset_index(drop=True)


def add_weather_features(
    df: pd.DataFrame,
    date_col: str = 'measured_at',
    copy: bool = False,
    location: Location = DEFAULT_LOCATION,
    features: List[str] = DEFAULT_FEATURES,
    weather_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Add weather features to a DataFrame by merging with weather data.
    
    Automatically handles both historical and future dates - fetches from
    historical API for past dates and forecast API for future dates.
    
    Arguments:
    - df: Input DataFrame
    - date_col: Name of the time column (default 'measured_at')
    - copy: Whether to copy the DataFrame before modifying (default False)
    - location: Tuple of (latitude, longitude) for weather data
    - features: List of weather features to fetch
    - weather_df: Optional pre-fetched weather DataFrame. If not provided,
                  weather will be fetched automatically based on the date range in df.
    
    Returns:
    - DataFrame with weather features merged
    """
    if copy:
        df = df.copy()
    
    # Convert date column to datetime if not already in this format
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            raise ValueError(f"Failed to convert '{date_col}' to datetime: {str(e)}")
    
    # Fetch weather data if not provided
    if weather_df is None:
        weather_df = fetch_weather_for_range(df[date_col], location, features)
    
    # Ensure weather_df has datetime column
    if 'time' not in weather_df.columns:
        raise ValueError("weather_df must have a 'time' column")
    
    if not pd.api.types.is_datetime64_any_dtype(weather_df['time']):
        weather_df['time'] = pd.to_datetime(weather_df['time'])
    
    # Round datetime to nearest hour for merging (weather data is hourly)
    df['_merge_time'] = df[date_col].dt.floor('h')
    weather_df['_merge_time'] = weather_df['time'].dt.floor('h')
    
    # Remove 'time' column from weather_df to avoid conflicts
    weather_cols = [col for col in weather_df.columns if col not in ['time', '_merge_time']]
    merge_df = weather_df[['_merge_time'] + weather_cols].drop_duplicates(subset=['_merge_time'])
    
    # Merge weather data
    df = df.merge(merge_df, on='_merge_time', how='left')
    
    # Clean up merge column
    df = df.drop(columns=['_merge_time'])
    
    return df
