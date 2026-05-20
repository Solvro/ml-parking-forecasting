import pandas as pd
import requests
from datetime import datetime, date
from typing import List, Optional, Tuple, Union

DateLike = Union[str, datetime, date, pd.Timestamp]


def _parse_date(d: DateLike, param_name: str) -> str:
    """
    Parse and validate a date input, returning 'YYYY-MM-DD' string.
    
    Arguments:
    - d: Date as string, datetime, date, or pd.Timestamp
    - param_name: Parameter name for error messages
    
    Returns:
    - Date string in 'YYYY-MM-DD' format
    
    Raises:
    - ValueError: If the date format is invalid
    """
    if isinstance(d, str):
        try:
            parsed = datetime.strptime(d, '%Y-%m-%d')
            return parsed.strftime('%Y-%m-%d')
        except ValueError:
            raise ValueError(
                f"Invalid date format for '{param_name}': '{d}'. Expected 'YYYY-MM-DD'."
            )
    elif isinstance(d, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(d).strftime('%Y-%m-%d')
    else:
        raise TypeError(
            f"'{param_name}' must be a string, datetime, date, or pd.Timestamp, got {type(d).__name__}"
        )

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Wrocław - Plac Grunwaldzki
DEFAULT_LOCATION = (51.1116, 17.0601)

DEFAULT_FEATURES = ["temperature_2m", "rain", "apparent_temperature", "snowfall", "snow_depth", "precipitation", "weather_code", "visibility", "surface_pressure", "wind_speed_10m"]

Location = Tuple[float, float]


def fetch_weather_forecast(
    location: Location = DEFAULT_LOCATION,
    features: List[str] = DEFAULT_FEATURES,
    forecast_days: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch weather forecast data from Open-Meteo API.
    
    Arguments:
    - location: Tuple of (latitude, longitude)
    - features: List of weather features to fetch
    - forecast_days: Number of forecast days to fetch (1-16). If None, uses API default (7 days).
    
    Returns:
    - DataFrame with weather forecast data
    """
    url = f"{FORECAST_URL}?latitude={location[0]}&longitude={location[1]}&hourly={','.join(features)}"
    if forecast_days is not None:
        url += f"&forecast_days={forecast_days}"
    response = requests.get(url, timeout=10)
    try:
        
        response.raise_for_status()
    
        df = pd.DataFrame(response.json()["hourly"])
        df["time"] = pd.to_datetime(df["time"])
    except requests.RequestException as e:
        print(f"Error fetching weather forecast: {e}")
    except KeyError as e:
        print(f"Unexpected response format: missing key {e}")
    return df


def fetch_historical_weather(
    start_date: DateLike,
    end_date: DateLike,
    location: Location = DEFAULT_LOCATION,
    features: List[str] = DEFAULT_FEATURES
) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo API.
    
    Arguments:
    - start_date: Start date (string 'YYYY-MM-DD', datetime, date, or pd.Timestamp)
    - end_date: End date (string 'YYYY-MM-DD', datetime, date, or pd.Timestamp)
    - location: Tuple of (latitude, longitude)
    - features: List of weather features to fetch
    
    Returns:
    - DataFrame with historical weather data
    
    Raises:
    - ValueError: If date format is invalid
    - TypeError: If date type is not supported
    """
    start_str = _parse_date(start_date, 'start_date')
    end_str = _parse_date(end_date, 'end_date')
    
    response = requests.get(
        f"{HISTORICAL_URL}?latitude={location[0]}&longitude={location[1]}"
        f"&start_date={start_str}&end_date={end_str}&hourly={','.join(features)}",
        timeout=30
    )
    try:
        response.raise_for_status()
        df = pd.DataFrame(response.json()["hourly"])
        df["time"] = pd.to_datetime(df["time"])
    except requests.RequestException as e:
        print(f"Error fetching historical weather data: {e}")
        return pd.DataFrame()
    except KeyError as e:
        print(f"Unexpected response format: missing key {e}")
        return pd.DataFrame()
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
    min_date = pd.Timestamp(dates.min())
    max_date = pd.Timestamp(dates.max())

    tz = min_date.tz
    if tz is not None:
        if max_date.tz is None:
            max_date = max_date.tz_localize(tz)
        today = pd.Timestamp.now(tz=tz).normalize()
    else:
        if max_date.tz is not None:
            max_date = max_date.tz_localize(None)
        today = pd.Timestamp.now().normalize()
    
    weather_dfs = []
    
    if min_date.date() < today.date():
        hist_end = min(max_date, today - pd.Timedelta(days=1))
        hist_df = fetch_historical_weather(
            min_date.strftime('%Y-%m-%d'),
            hist_end.strftime('%Y-%m-%d'),
            location,
            features
        )
        weather_dfs.append(hist_df)
    
    if max_date.date() >= today.date():
        forecast_days = (max_date.date() - today.date()).days + 1
        forecast_days = max(1, min(forecast_days, 16))
        forecast_df = fetch_weather_forecast(location, features, forecast_days)
        weather_dfs.append(forecast_df)
    
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
    - copy: Whether to copy DataFrames before modifying (default False).
            Applies to both df and weather_df to prevent mutation of inputs.
    - location: Tuple of (latitude, longitude) for weather data
    - features: List of weather features to fetch
    - weather_df: Optional pre-fetched weather DataFrame. If not provided,
                  weather will be fetched automatically based on the date range in df.
    
    Returns:
    - DataFrame with weather features merged
    """
    if copy:
        df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            raise ValueError(f"Failed to convert '{date_col}' to datetime: {str(e)}")
    
    if weather_df is None:
        weather_df = fetch_weather_for_range(df[date_col], location, features)
    elif copy:
        weather_df = weather_df.copy()
    
    if 'time' not in weather_df.columns:
        raise ValueError("weather_df must have a 'time' column")
    
    if not pd.api.types.is_datetime64_any_dtype(weather_df['time']):
        weather_df['time'] = pd.to_datetime(weather_df['time'])
    
    merge_time_df = pd.to_datetime(df[date_col], utc=True).dt.floor('h').dt.tz_localize(None)
    merge_time_weather = pd.to_datetime(weather_df['time'], utc=True).dt.floor('h').dt.tz_localize(None)

    left_df = df.assign(_merge_time=merge_time_df)
    right_df = weather_df.assign(_merge_time=merge_time_weather)


    weather_cols = [col for col in right_df.columns if col not in ['time', '_merge_time']]
    merge_df = right_df[['_merge_time'] + weather_cols].drop_duplicates(subset=['_merge_time'])

    df = left_df.merge(merge_df, on='_merge_time', how='left')

    df = df.drop(columns=['_merge_time'])
    
    return df

df = pd.read_parquet("data/Data/parking_availabilities.parquet")
print(df.head())
df_with_weather = add_weather_features(df, date_col='measured_at', location=DEFAULT_LOCATION, features=DEFAULT_FEATURES)
df_with_weather.to_csv("data/parkings_with_weather.csv", index=False)