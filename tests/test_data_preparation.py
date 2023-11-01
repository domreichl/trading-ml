import pandas as pd
import datetime as dt

from utils.data_preparation import download_data, impute_missing_data, trim_time_series


def get_weekdays(start_date: dt.datetime, end_date: dt.datetime) -> list:
    weekdays = []
    day = start_date
    while day <= end_date:
        if day.weekday() not in [5, 6]:
            weekdays.append(str(day.date()))
        day += dt.timedelta(days=1)
    return weekdays


def test_data_preparation():
    start_date = dt.datetime(2023, 10, 1)  # Sunday
    end_date = dt.datetime(2023, 10, 31)  # Tuesday
    holidays = ["2023-10-26"]  # Nationalfeiertag
    weekdays = (
        pd.date_range(start=start_date, end=end_date, freq="B").astype(str).tolist()
    )
    assert weekdays == get_weekdays(start_date, end_date)

    securities = {
        "AT0000743059": "omv-ag-AT0000743059",
        "AT0000937503": "voestalpine-ag-AT0000937503",
    }
    df = download_data(start_date, end_date, securities)
    assert len(weekdays) == df["Date"].nunique() + len(holidays)
    assert list(set(weekdays).difference(set(df["Date"].astype(str))))[0] == holidays[0]

    weekdays_before_first_monday = []
    weekdays_after_last_friday = ["2023-10-30", "2023-10-31"]
    df = pd.concat(
        [
            trim_time_series(impute_missing_data(df[df["ISIN"] == isin], weekdays))
            for isin in df["ISIN"].unique()
        ]
    )
    assert (
        len(weekdays)
        - len(weekdays_before_first_monday)
        - len(weekdays_after_last_friday)
        == df["Date"].nunique()
    )
    assert set(weekdays).difference(set(df["Date"].astype(str))) == set(
        weekdays_before_first_monday + weekdays_after_last_friday
    )
    for isin in df["ISIN"].unique():
        assert (
            df[(df["ISIN"] == isin) & (df["Date"].astype(str) == "2023-10-25")][
                "Close"
            ].iloc[0]
            == df[(df["ISIN"] == isin) & (df["Date"].astype(str) == holidays[0])][
                "Close"
            ].iloc[0]
        )
