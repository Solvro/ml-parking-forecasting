import pandas as pd
import re

location_regex_list = [
    r"\b[a-h]-?\d{1,2}\.?\d?\b",
    r"\bsks\b",
    r"\bpwr\b",
    r"janiszewskiego",
    r"wrońskiego",
    r"wyspiańskiego",
    r"politechnika\s+wrocławska",
    r"nowa sala senatu",
    r"grunwaldzki",
    r"strefa kultury studenckiej",
    r"geocentrum"
]

location_pattern = re.compile(
    "|".join(location_regex_list),
    re.IGNORECASE,
)

def filter_event_locations(df_events, pattern=location_pattern, copy: bool = False):
    """Creates a boolean column "location_filtered" in the input dataframe df_events,
    indicating whether the "location" column matches any of the specified regex patterns."""

    if copy:
        df_events = df_events.copy()

    df_events["is_location_filtered"] = df_events["location"].str.contains(pattern, na=False)

    return df_events


summary_regex_list = [
    r"wieczór gier planszowych",
    r"witkon",
    r"śniadanie z rektorem",
    r"drzwi otwarte",
    r"(?<!rejs\s)aktywności",
    r"targi pracy",
    r"i love pwr",
    r"spotkanie choinkowe",
    r"among pwr"]

summary_pattern = re.compile(
    "|".join(summary_regex_list),
    re.IGNORECASE,
)


def filter_event_summaries(df_events, pattern=summary_pattern, copy: bool = False):
    """Creates a boolean column "summary_filtered" in the input dataframe df_events,
    indicating whether the "summary" column matches any of the specified regex patterns."""

    if copy:
        df_events = df_events.copy()
    df_events["is_summary_filtered"] = df_events["summary"].str.contains(pattern, na=False)
    return df_events


def filter_events(df_events, location_pattern=location_pattern, summary_pattern=summary_pattern, copy: bool = False):
    """
    Creates a single boolean column 'filter_events' which is True
    if either location or summary matches the specified regex patterns.
    Drops intermediate helper columns.
    """

    if copy:
        df_events = df_events.copy()

    df_events = filter_event_locations(df_events, location_pattern)
    df_events = filter_event_summaries(df_events, summary_pattern)

    df_events["is_filtered_event"] = (df_events["is_location_filtered"] | df_events["is_summary_filtered"])

    df_events = df_events.drop(columns=["is_location_filtered", "is_summary_filtered"])

    return df_events


if __name__ == "__main__":
    df_events = pd.read_parquet('data/pwr_events.parquet')
    #df_filtered_locations = filter_event_locations(df_events)
    #df_filtered = filter_event_summaries(df_filtered_locations)
    #print(df_filtered[["location", "is_location_filtered", "summary", "is_summary_filtered"]].head(20))

    df_filtered = filter_events(df_events)
    print(df_filtered[["location", "summary", "is_filtered_event"]].head(20))