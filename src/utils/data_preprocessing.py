import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import Union

from utils.config import Config
from utils.data_classes import MultipleTimeSeries
from utils.file_handling import DataHandler


class DataPreprocessor:
    def __init__(
        self,
        df: pd.DataFrame,
        look_back_window_size: int,
        test_days: int,
        normalize: bool,
    ):
        self.df = df
        self.dates = list(self.df["Date"].unique())
        self.securities = list(self.df["ISIN"].unique())
        self.ts_count = len(self.securities)
        self.total_days = len(self.df) // self.ts_count
        self.look_back_window_size = look_back_window_size
        self.time_steps = self.total_days - self.look_back_window_size
        self.test_days = test_days
        self.train_days = self.time_steps - self.test_days
        self.compute_log_returns()
        self.prepare_features()
        self.split_train_test()
        if normalize:
            self.scaler = self.normalize()
        else:
            self.scaler = None
        self.mts = self.create_mts()

    def compute_log_returns(self) -> None:
        self.df["LogReturn"] = pd.concat(
            [
                np.log(
                    self.df[self.df["ISIN"] == ISIN]["Close"]
                    / self.df[self.df["ISIN"] == ISIN]["Close"].shift(1)
                )
                for ISIN in self.securities
            ]
        )
        self.df["LogReturn"] = self.df["LogReturn"].fillna(0.0)

    def prepare_features(self) -> None:
        x_dict, y_dict = {}, {}
        self.log_returns, self.close_prices = {}, {}
        for isin in self.securities:
            df_isin = self.df[self.df["ISIN"] == isin]
            self.close_prices[isin] = df_isin["Close"].values
            self.log_returns[isin] = df_isin["LogReturn"].values
            x_dict[isin], y_dict[isin] = [], []
            for i in range(self.look_back_window_size, len(df_isin) - self.test_days):
                x_dict[isin].append(
                    self.log_returns[isin][i - self.look_back_window_size : i]
                )
                y_dict[isin].append(self.log_returns[isin][i : i + self.test_days])
        self.x = np.stack([np.array(features) for features in x_dict.values()], 2)
        self.y = np.stack([np.array(labels) for labels in y_dict.values()], 2)
        assert self.x.shape == (
            self.train_days,
            self.look_back_window_size,
            len(self.securities),
        )
        assert self.y.shape == (self.train_days, self.test_days, len(self.securities))

    def split_train_test(self) -> None:
        self.x_train, self.x_test = (
            self.x[:-1, :, :],
            self.x[-1, :, :],
        )
        self.y_train, self.y_test = (
            self.y[:-1, :, :],
            self.y[-1, :, :],
        )

    def normalize(self) -> MinMaxScaler:
        scaler = MinMaxScaler()
        self.x_train = scaler.fit_transform(self.x_train.reshape(-1, 1)).reshape(
            self.train_days - 1, self.look_back_window_size, len(self.securities)
        )
        self.x_test = scaler.transform(self.x_test.reshape(-1, 1)).reshape(
            self.look_back_window_size, len(self.securities)
        )
        self.y_train = scaler.transform(self.y_train.reshape(-1, 1)).reshape(
            self.train_days - 1, self.test_days, len(self.securities)
        )
        self.y_test = scaler.transform(self.y_test.reshape(-1, 1)).reshape(
            self.test_days, len(self.securities)
        )
        return scaler

    def create_mts(self) -> MultipleTimeSeries:
        return MultipleTimeSeries(
            self.dates,
            self.securities,
            self.log_returns,
            self.close_prices,
            self.x_train,
            self.y_train,
            self.x_test,
            self.y_test,
            self.scaler,
        )

    def get_mts(self) -> MultipleTimeSeries:
        return self.mts


def preprocess_data(
    csv_file: Union[str, Path],
    model_name: str = "",
    look_back_window_size: int = 260,
    test_days: int = 10,
    normalize: bool = True,
) -> MultipleTimeSeries:
    df = DataHandler().load_csv_data(csv_file)
    if model_name:
        lbws = Config().models[model_name]["look_back_window_size"]
    else:
        lbws = look_back_window_size
    return DataPreprocessor(df, lbws, test_days, normalize).get_mts()
