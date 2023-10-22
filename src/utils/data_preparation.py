import urllib.request
import datetime as dt
import pandas as pd
import numpy as np

from utils.config import Config
from utils.file_handling import DataHandler


def prepare_data(csv_name: str, cfg: Config):
    df = download_data(cfg.start_date, cfg.end_date, cfg.securities)
    df = impute_missing_data(df, get_weekdays(cfg.start_date, cfg.end_date))
    DataHandler().write_csv_data(df, csv_name)


def download_data(
    start_date: dt.datetime, end_date: dt.datetime, securities: dict
) -> pd.DataFrame:
    dfs = []
    columns = ["ISIN", "Date", "Close"]
    for isin, security in securities.items():
        date_format = "%d.%m.%Y"
        if security == "ATX":
            url = f"https://www.wienerborse.at/indizes/aktuelle-indexwerte/historische-daten/?ISIN=AT0000999982&ID_NOTATION=92866&c7012%5BDATETIME_TZ_END_RANGE%5D={end_date.strftime(date_format)}&c7012%5BDATETIME_TZ_START_RANGE%5D={start_date.strftime(date_format)}&c7012%5BDOWNLOAD%5D=csv"
        else:
            url = f"https://www.wienerborse.at/aktien-prime-market/{security}/historische-daten/?c48840%5BDOWNLOAD%5D=csv&c48840%5BDATETIME_TZ_END_RANGE%5D={end_date.strftime(date_format)}&c48840%5BDATETIME_TZ_START_RANGE%5D={start_date.strftime(date_format)}T00%3A00%3A00%2B01%3A00"
        with urllib.request.urlopen(url) as csv_file:
            df = pd.read_csv(csv_file, sep=";")
            df["ISIN"] = isin
            df["Date"] = pd.to_datetime(df["Datum"], format=date_format)
            df["Close"] = df["Schlusspreis"].str.replace(",", ".").astype(float)
            dfs.append(df[columns])
    return pd.concat(dfs)


def get_weekdays(start_date: dt.datetime, end_date: dt.datetime) -> list:
    """builds a list of all weekdays (Mon-Fri) within a given date range"""

    weekdays = []
    day = start_date
    while day <= end_date:
        if day.weekday() not in [5, 6]:
            weekdays.append(str(day.date()))
        day += dt.timedelta(days=1)
    print(
        f"Date range from {start_date.date()} to {end_date.date()} has {len(weekdays)} weekdays."
    )
    return weekdays


def impute_missing_data(df: pd.DataFrame, weekdays: list) -> pd.DataFrame:
    """fills in missing close prices by imputing the previous day's value;
    ensures that the time series can be separated evenly into 5-day weekly cycles"""

    dfs = []
    for isin in df["ISIN"].unique():
        df_isin = df[df["ISIN"] == isin]

        # find and impute missing values
        missing_dates = set(pd.to_datetime(pd.Series(weekdays)).unique()).difference(
            set(df_isin["Date"].unique())
        )
        print(
            f"ISIN {isin} has {len(df_isin)} entries: {len(missing_dates)} values will be imputed."
        )
        df_isin = pd.concat(
            [
                df_isin,
                pd.DataFrame(
                    {"ISIN": isin, "Date": list(missing_dates), "Close": np.nan}
                ),
            ]
        )
        df_isin.sort_values(by="Date", inplace=True)
        while len(df_isin[df_isin["Close"].isna()]) > 0:
            df_isin["Close"] = df_isin["Close"].fillna(
                (df_isin["Close"].shift(1))
            )  # impute prev value

        # make the series start on a Monday and end on a Friday
        weekday = df_isin["Date"].iloc[0].weekday()
        while weekday != 0:  # Monday
            df_isin = df_isin.iloc[1:]
            weekday = df_isin["Date"].iloc[0].weekday()
        weekday = df_isin["Date"].iloc[-1].weekday()
        while weekday != 4:  # Friday
            df_isin = df_isin.iloc[:-1]
            weekday = df_isin["Date"].iloc[-1].weekday()

        dfs.append(df_isin)

    return pd.concat(dfs)