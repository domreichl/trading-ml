import urllib.request
import datetime as dt
import pandas as pd
import numpy as np

from utils.config import Config
from utils.file_handling import DataHandler


def prepare_data(csv_name: str, cfg: Config):
    df = download_data(cfg.start_date, cfg.end_date, cfg.securities)
    weekdays = (
        pd.date_range(start=cfg.start_date, end=cfg.end_date, freq="B")
        .astype(str)
        .tolist()
    )
    df = pd.concat(
        [
            trim_time_series(impute_missing_data(df[df["ISIN"] == isin], weekdays))
            for isin in df["ISIN"].unique()
        ]
    )
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


def impute_missing_data(df: pd.DataFrame, weekdays: list) -> pd.DataFrame:
    missing_dates = list(
        set(pd.to_datetime(pd.Series(weekdays)).unique()).difference(
            set(df["Date"].unique())
        )
    )
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {"ISIN": df["ISIN"].iloc[0], "Date": missing_dates, "Close": np.nan}
            ),
        ]
    )
    df.sort_values(by="Date", inplace=True)
    df["Close"] = df["Close"].ffill()
    return df


def trim_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """make the series start on a Monday and end on a Friday so that it can be separated evenly into 5-day weekly cycles"""

    weekday = df["Date"].iloc[0].weekday()
    while weekday != 0:  # Monday
        df = df.iloc[1:]
        weekday = df["Date"].iloc[0].weekday()
    weekday = df["Date"].iloc[-1].weekday()
    while weekday != 4:  # Friday
        df = df.iloc[:-1]
        weekday = df["Date"].iloc[-1].weekday()
    return df
