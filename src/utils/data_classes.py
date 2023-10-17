import numpy as np
import pandas as pd
import datetime as dt
from dataclasses import dataclass

from config.data_config import data_config


@dataclass
class MultipleTimeSeries:
    dates: list[str]
    names: list[str]
    log_returns: dict
    close_prices: dict
    x_train: np.array
    y_train: np.array
    x_test: np.array
    y_test: np.array
    scaler: object

    def get_returns_from_features(self, features: np.array) -> np.array:
        if self.scaler:
            log_returns = np.squeeze(
                self.scaler.inverse_transform(features.reshape(-1, 1))
            ).reshape(features.shape)
        else:
            log_returns = features
        return np.exp(log_returns)

    def get_returns_dict_from_features(self, features: np.array) -> dict:
        return {
            ISIN: np.round(self.get_returns_from_features(features)[:, i], 8).tolist()
            for i, ISIN in enumerate(self.names)
        }

    def get_forecast_dates(
        self, date_format: str = data_config["date_format"]
    ) -> list[str]:
        day = dt.datetime.strptime(self.dates[-1], date_format)
        if day.weekday() == 4:  # Fri -> Mon
            start = day + dt.timedelta(days=3)
        elif day.weekday() in [5, 6]:
            raise Exception("Error: Dates contain a Saturday or Sunday.")
        else:
            start = day + dt.timedelta(days=1)
        forecast_dates = [start.strftime(date_format)]
        next_day = start + dt.timedelta(days=1)
        while len(forecast_dates) < len(self.y_test):
            if next_day.weekday() not in [5, 6]:
                forecast_dates.append(next_day.strftime(date_format))
            next_day += dt.timedelta(days=1)
        return forecast_dates

    def merge_features(self, for_deep_learning: bool = False) -> None:
        self.x_train = np.concatenate([self.x_train, np.expand_dims(self.x_test, 0)], 0)
        self.y_train = np.concatenate([self.y_train, np.expand_dims(self.y_test, 0)], 0)
        if not for_deep_learning:
            self.x_train = np.concatenate(
                [self.x_train[:, len(self.y_test) :, :], self.y_train], 1
            )
            self.y_train = self.x_test = np.array([])

    def get_train_df(self, idx: int = -1, ts_name: str = "") -> pd.DataFrame():
        if ts_name:
            for i, name in enumerate(self.names):
                if name == ts_name:
                    ts_dict = {name: self.x_train[idx, :, i]}
                    break
        else:
            ts_dict = {
                name: self.x_train[idx, :, i] for i, name in enumerate(self.names)
            }
        dates = self.dates[
            -self.x_train.shape[1] - len(self.y_test) : -len(self.y_test)
        ]
        df = pd.DataFrame(ts_dict, index=dates).unstack().reset_index()
        df.columns = ["unique_id", "ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])
        df.sort_values("ds", inplace=True)
        return df

    def get_test_prices(self) -> dict:
        return {
            name: np.round(self.close_prices[name][-len(self.y_test) :], 2).tolist()
            for name in self.names
        }

    def get_test_returns(self) -> dict:
        return {
            name: np.exp(log_returns[-len(self.y_test) :])
            for name, log_returns in self.log_returns.items()
        }

    def get_test_dates(self) -> list:
        return self.dates[-len(self.y_test) :]

    def get_naive_errors(self) -> tuple[float, float]:
        # MAE and RMSE of a naive model where y_t = y_t-1
        naive_mae = float(np.mean(np.abs(np.diff(self.get_eval_returns(), axis=0))))
        naive_rmse = float(
            np.sqrt(np.mean(np.square(np.diff(self.get_eval_returns(), axis=0))))
        )
        return naive_mae, naive_rmse

    def get_eval_returns(self) -> np.array:
        return np.concatenate(
            [
                self.get_returns_from_features(self.x_train[-1]),
                self.get_returns_from_features(self.y_test),
            ],
            0,
        )
