from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

#custom trasformers (methods) 
from feature_engineering.holiday_feature import add_holiday_features
from feature_engineering.date_features import add_date_features

#scikit-learn compatble transformers
from preprocessing.parking_resampler import ParkingResampler
from preprocessing.parking_sks_merger import ParkingSKSMerger
from preprocessing.parking_calendar_merger import ParkingCalendarMerger
from preprocessing.parking_imputer import ParkingImputer
from feature_engineering.parking_capacity_corrector import ParkingSpacesCorrector
from feature_engineering.day_features import DayFeaturesCreator
from feature_engineering.rolling_features_creator import RollingFeaturesCreator
from feature_engineering.expanding_features_creator import ExpandingFeaturesCreator
from feature_engineering.lag_features_creator import LagFeaturesCreator
from feature_engineering.cannibalisation_adder import CannibalisationFeaturesCreator

#Constants for feature engineering
CONST_FREQUENCY_MINUTES = 5
CONST_CONVERT_32_BIT_VALUES = True
CONST_SHIFT_GUARD_PERIODS = 15
CONST_CANNIVALISATION_GUARD_PERIODS = 15
CONST_COPY = False

# Dictionaries with config 
agg_rules = {
        'spaces_left': 'mean',
        #'trend': 'mean', # this trend is weird.. i still have no idea how to use it seems semi random at times 
    }

expansions = {
    "Parking Wrońskiego": '2025-05-21',
    #Others are ommited so they will be treated as never expanded (add them if they were expanded)
}
size_before_expansion ={
    "Parking Wrońskiego": 192,
    #Others are ommited so they will be treated as never expanded (add them if they were expanded)
}
# hour every 12 records (1 hour = 12 records at 5 min frequency, 7 days = 288 records per day * 7)
rolling_features = {
    # Key : (Suffix,Type, Window Size, List of Columns)
    # Avaialvble types: "delta", "mean", "std", "max", "min"
    1: ("delta_1h", "delta", 12, ["spaces_left"]), 
    2: ("Rolling_mean_1d", "mean", 288, ["spaces_left"]), 
    3: ("Rolling_mean_7d", "mean", 288*7, ["spaces_left"]),
    4: ("Rolling_std_1d", "std", 288, ["spaces_left"]),
    5: ("Rolling_std_7d", "std", 288*7, ["spaces_left"]),
    6: ("Rolling_max_7d", "max", 288*7, ["spaces_left"]),
    7: ("Rolling_min_7d", "min", 288*7, ["spaces_left"]),
}

expanding_features = {
    # Key : (Suffix, Type, List of Columns)
    1: ("cumulative_mean", "mean", ["spaces_left"]),
    2: ("cumulative_max", "max", ["spaces_left"]),
    3: ("cumulative_min", "min", ["spaces_left"]),
    4: ("cumulative_trend", "std", ["spaces_left"]),
}

lag_features = {
    # Key : (Suffix, Window Size, List of Columns)
    1: ("lag_1h", 12, ["spaces_left", "utilization_rate"]), 
    2: ("lag_7d", 288*7, ["spaces_left", "utilization_rate", "spaces_left_Rolling_mean_7d", "spaces_left_Rolling_std_7d", "spaces_left_Rolling_max_7d", "spaces_left_Rolling_min_7d"]), 
}

cannibalistion_features = {
    # Key : (Suffix, Type, List of Columns)
    1: ("cann_closest_parking", "closest", ["spaces_left", "utilization_rate", "spaces_left_lag_7d", "utilization_rate_lag_7d", "spaces_left_delta_1h"]),
    2: ("cann_second_closest_parking", "second_closest", ["spaces_left", "utilization_rate"]),
    3: ("cann_third_closest_parking", "third_closest", ["spaces_left", "utilization_rate"]),
    4: ("cann_fourth_closest_parking", "fourth_closest", ["spaces_left", "utilization_rate"]),
}

def build_feature_pipeline(df_sks_users, df_parkings, df_calendar):
    #Prep transformers
    parking_resampler = ParkingResampler(convert_to_32=CONST_CONVERT_32_BIT_VALUES, agg_rules=agg_rules)

    #use kw_args to pass additional arguments to the function transformer
    holiday_transformer = FunctionTransformer(add_holiday_features
    , kw_args={'copy': CONST_COPY})

    date_transformer = FunctionTransformer(add_date_features
    , kw_args={'copy': CONST_COPY})

    parking_sks_merger = ParkingSKSMerger(sks_df=df_sks_users, parkings_df=df_parkings, freq_minutes=CONST_FREQUENCY_MINUTES, convert_to_32=CONST_CONVERT_32_BIT_VALUES, copy=CONST_COPY)
    parking_calendar_merger = ParkingCalendarMerger(calendar_df=df_calendar, freq_minutes=CONST_FREQUENCY_MINUTES, convert_to_32=CONST_CONVERT_32_BIT_VALUES, copy=CONST_COPY)
    parking_imputer = ParkingImputer(cols_to_impute=['spaces_left','active_users','sks_dist_to_users_ratio'], freq_minutes=CONST_FREQUENCY_MINUTES, ffill_limit=60, convert_to_32=CONST_CONVERT_32_BIT_VALUES, copy=CONST_COPY)
    parkings_capacity_corrector = ParkingSpacesCorrector(df_parkings, expansions, size_before_expansion, convert_to_32=CONST_CONVERT_32_BIT_VALUES, copy=CONST_COPY)
    day_feature_creator = DayFeaturesCreator(df_parkings, convert_to_32=CONST_CONVERT_32_BIT_VALUES, copy=CONST_COPY)
    rolling_features_creator = RollingFeaturesCreator(rolling_features =rolling_features, convert_to_32=CONST_CONVERT_32_BIT_VALUES, shift_guard_periods=CONST_SHIFT_GUARD_PERIODS, copy=CONST_COPY)
    lag_features_creator = LagFeaturesCreator(lag_features, convert_to_32=CONST_CONVERT_32_BIT_VALUES, copy=CONST_COPY)
    cannibalistion_features_creator = CannibalisationFeaturesCreator(df_parkings, cannibalistion_features, cannibalisation_guard=CONST_CANNIVALISATION_GUARD_PERIODS, convert_to_32=CONST_CONVERT_32_BIT_VALUES, copy=CONST_COPY)

    #Transformer pipline
    feature_pipeline = Pipeline([
        ('resampler', parking_resampler),
        ('date_features', date_transformer),       
        ('holiday_features', holiday_transformer), 
        ('sks_merger', parking_sks_merger),
        ('calendar_merger', parking_calendar_merger),
        ('imputer', parking_imputer),
        ('capacity_corrector', parkings_capacity_corrector),
        ('day_features', day_feature_creator),
        ('rolling_features', rolling_features_creator),
        ('lag_features', lag_features_creator),
        ('cannibalisation_features', cannibalistion_features_creator)
    ])
    return feature_pipeline
