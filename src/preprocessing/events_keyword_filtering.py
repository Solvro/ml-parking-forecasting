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

location_combined_regex = "|".join(location_regex_list)


def filter_event_locations(df_events, location_regex=location_combined_regex):
    """Creates a boolean column "location_filtered" in the input dataframe df_events,
    indicating whether the "location" column matches any of the specified regex patterns."""

    df_events = df_events.copy()
    df_events["location_filtered"] = df_events["location"].str.contains(
        location_regex, case=False, regex=True, na=False
    )
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

summary_combined_regex = "|".join(summary_regex_list)


def filter_event_summaries(df_events, summary_regex=summary_combined_regex):
    """Creates a boolean column "summary_filtered" in the input dataframe df_events,
    indicating whether the "summary" column matches any of the specified regex patterns."""

    df_events = df_events.copy()
    df_events["summary_filtered"] = df_events["summary"].str.contains(
        summary_regex, case=False, regex=True, na=False
    )
    return df_events


def filter_events(df_events, location_regex=location_combined_regex, summary_regex=summary_combined_regex):
    """
    Creates a single boolean column 'filter_events' which is True
    if either location or summary matches the specified regex patterns.
    Drops intermediate helper columns.
    """
    df_events = df_events.copy()

    df_events = filter_event_locations(df_events, location_regex)
    df_events = filter_event_summaries(df_events, summary_regex)

    df_events["filter_events"] = (df_events["location_filtered"] | df_events["summary_filtered"])

    df_events = df_events.drop(columns=["location_filtered", "summary_filtered"])

    return df_events


if __name__ == "__main__":
    import pandas as pd

    df_events = pd.read_parquet('data/pwr_events.parquet')
    #df_filtered_locations = filter_event_locations(df_events)
    #df_filtered = filter_event_summaries(df_filtered_locations)
    #print(df_filtered[["location", "location_filtered", "summary", "summary_filtered"]].head(20))

    df_filtered = filter_events(df_events)
    print(df_filtered[["location", "summary", "filter_events"]].head(20))