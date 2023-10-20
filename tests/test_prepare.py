import datetime as dt

from utils.data_preparation import download_data, impute_missing_data, get_weekdays


test_config = {
    "start_date": dt.datetime(2023, 9, 18),  # Monday
    "end_date": dt.datetime(2023, 9, 22),  # Friday
}


def test_data_preparation_wiener_boerse():
    df = download_data(
        test_config["start_date"],
        test_config["end_date"],
        {
            "AT0000743059": "omv-ag-AT0000743059",
            "AT0000937503": "voestalpine-ag-AT0000937503",
        },
    )
    assert len(df) == 2 * 5
    assert len(df.columns) == 3
    date_to_impute = dt.datetime(2023, 9, 21)
    df = df[df["Date"] != date_to_impute]
    assert len(df[df["Date"] == date_to_impute]) == 0
    df = impute_missing_data(
        df, get_weekdays(test_config["start_date"], test_config["end_date"])
    )
    assert len(df[df["Date"] == date_to_impute]) == 2
