from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.feature_engineering.holiday_feature import add_holiday_features
from src.feature_engineering.date_features import add_date_features
from src.preprocessing.parking_resampler import ParkingResampler
from src.preprocessing.parking_sks_merger import ParkingSKSMerger
from src.preprocessing.parking_calendar_merger import ParkingCalendarMerger
from src.preprocessing.parking_imputer import ParkingImputer
from src.feature_engineering.parking_capacity_corrector import ParkingSpacesCorrector
from src.feature_engineering.day_features import DayFeaturesCreator
from src.feature_engineering.rolling_features_creator import RollingFeaturesCreator
from src.feature_engineering.expanding_features_creator import ExpandingFeaturesCreator
from src.feature_engineering.lag_features_creator import LagFeaturesCreator
from src.feature_engineering.cannibalisation_adder import CannibalisationFeaturesCreator

@dataclass
class FeaturePipelineConfig:
    """All knobs for the feature pipeline. Round-trips through JSON."""

    #merge / resample rules
    agg_rules: dict = field(default_factory=lambda: {
        'spaces_left': 'mean',
        # 'trend': 'mean',  # weird, semi-random — leave out for now
    })

    # parking lot expansions 
    expansions: dict = field(default_factory=lambda: {
        "Parking Wrońskiego": '2025-05-21',
    })
    size_before_expansion: dict = field(default_factory=lambda: {
        "Parking Wrońskiego": 192,
    })

    # feature dicts: {id: (suffix, type, window, [cols])}
    # 1 hour = 12 records @ 5min freq; 1 day = 288; 7 days = 288*7
    rolling_features: dict = field(default_factory=lambda: {
        1: ("delta_1h",        "delta", 12,      ["spaces_left"]),
        2: ("Rolling_mean_1d", "mean",  288,     ["spaces_left"]),
        3: ("Rolling_mean_7d", "mean",  288 * 7, ["spaces_left"]),
        4: ("Rolling_std_1d",  "std",   288,     ["spaces_left"]),
        5: ("Rolling_std_7d",  "std",   288 * 7, ["spaces_left"]),
        6: ("Rolling_max_7d",  "max",   288 * 7, ["spaces_left"]),
        7: ("Rolling_min_7d",  "min",   288 * 7, ["spaces_left"]),
    })

    expanding_features: dict = field(default_factory=lambda: {
        1: ("cumulative_mean",  "mean", ["spaces_left"]),
        2: ("cumulative_max",   "max",  ["spaces_left"]),
        3: ("cumulative_min",   "min",  ["spaces_left"]),
        4: ("cumulative_trend", "std",  ["spaces_left"]),
    })

    lag_features: dict = field(default_factory=lambda: {
        1: ("lag_1h", 12, ["spaces_left", "utilization_rate"]),
        2: ("lag_7d", 288 * 7, [
            "spaces_left", "utilization_rate",
            "spaces_left_Rolling_mean_7d", "spaces_left_Rolling_std_7d",
            "spaces_left_Rolling_max_7d",  "spaces_left_Rolling_min_7d",
        ]),
    })

    cannibalisation_features: dict = field(default_factory=lambda: {
        1: ("cann_closest_parking",        "closest",        ["spaces_left", "spaces_left_lag_7d", "utilization_rate_lag_1h", "utilization_rate_lag_7d", "spaces_left_delta_1h"]),
        2: ("cann_second_closest_parking", "second_closest", ["spaces_left", "utilization_rate_lag_1h"]),
        3: ("cann_third_closest_parking",  "third_closest",  ["spaces_left", "utilization_rate_lag_1h"]),
        4: ("cann_fourth_closest_parking", "fourth_closest", ["spaces_left", "utilization_rate_lag_1h"]),
    })

    frequency_minutes: int = 5
    convert_to_32_bit: bool = True
    shift_guard_periods: int = 15
    cannibalisation_guard_periods: int = 15
    copy: bool = False
    ffill_limit: int = 60
    cols_to_impute: list = field(default_factory=lambda:
        ['spaces_left', 'active_users', 'sks_dist_to_users_ratio']
    )

    # --- (de)serialization -------------------------------------------
    # Feature dicts that need int keys + tuple values restored after JSON load.
    _INDEXED_FEATURE_DICTS = (
        'rolling_features',
        'expanding_features',
        'lag_features',
        'cannibalisation_features',
    )

    @classmethod
    def from_json(cls, path: str | Path) -> "FeaturePipelineConfig":
        with open(path, 'r', encoding='utf-8') as f:
            data: dict[str, Any] = json.load(f)

        # JSON has no int keys and no tuples — rebuild them.
        for key in cls._INDEXED_FEATURE_DICTS:
            if key in data and data[key] is not None:
                data[key] = {int(k): tuple(v) for k, v in data[key].items()}

        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)



# Builder

def build_feature_pipeline(
    config: FeaturePipelineConfig,
    df_sks_users,
    df_parkings,
    df_calendar,
) -> Pipeline:
    # Prep transformers
    parking_resampler = ParkingResampler(
        convert_to_32=config.convert_to_32_bit,
        agg_rules=config.agg_rules,
    )

    # use kw_args to pass additional arguments to the function transformer
    holiday_transformer = FunctionTransformer(
        add_holiday_features, kw_args={'copy': config.copy}
    )
    date_transformer = FunctionTransformer(
        add_date_features, kw_args={'copy': config.copy}
    )

    parking_sks_merger = ParkingSKSMerger(
        sks_df=df_sks_users,
        parkings_df=df_parkings,
        freq_minutes=config.frequency_minutes,
        convert_to_32=config.convert_to_32_bit,
        copy=config.copy,
    )
    parking_calendar_merger = ParkingCalendarMerger(
        calendar_df=df_calendar,
        freq_minutes=config.frequency_minutes,
        convert_to_32=config.convert_to_32_bit,
        copy=config.copy,
    )
    parking_imputer = ParkingImputer(
        cols_to_impute=config.cols_to_impute,
        freq_minutes=config.frequency_minutes,
        ffill_limit=config.ffill_limit,
        convert_to_32=config.convert_to_32_bit,
        copy=config.copy,
    )
    parkings_capacity_corrector = ParkingSpacesCorrector(
        df_parkings,
        config.expansions,
        config.size_before_expansion,
        convert_to_32=config.convert_to_32_bit,
        copy=config.copy,
    )
    day_feature_creator = DayFeaturesCreator(
        df_parkings,
        convert_to_32=config.convert_to_32_bit,
        copy=config.copy,
    )
    rolling_features_creator = RollingFeaturesCreator(
        rolling_features=config.rolling_features,
        convert_to_32=config.convert_to_32_bit,
        shift_guard_periods=config.shift_guard_periods,
        copy=config.copy,
    )
    lag_features_creator = LagFeaturesCreator(
        config.lag_features,
        convert_to_32=config.convert_to_32_bit,
        copy=config.copy,
    )
    cannibalisation_features_creator = CannibalisationFeaturesCreator(
        df_parkings,
        config.cannibalisation_features,
        cannibalisation_guard=config.cannibalisation_guard_periods,
        convert_to_32=config.convert_to_32_bit,
        copy=config.copy,
    )

    # Transformer pipeline
    feature_pipeline = Pipeline([
        ('resampler',                parking_resampler),
        ('date_features',            date_transformer),
        ('holiday_features',         holiday_transformer),
        ('sks_merger',               parking_sks_merger),
        ('calendar_merger',          parking_calendar_merger),
        ('imputer',                  parking_imputer),
        ('capacity_corrector',       parkings_capacity_corrector),
        ('day_features',             day_feature_creator),
        ('rolling_features',         rolling_features_creator),
        ('lag_features',             lag_features_creator),
        ('cannibalisation_features', cannibalisation_features_creator),
    ])
    return feature_pipeline
