import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from window_ops.rolling import rolling_mean, rolling_max, rolling_min
from xgboost import XGBRegressor

from config.model_config import model_config
from utils.data_classes import MultipleTimeSeries
from utils.evaluation import evaluate_return_predictions


def load_lightgbm() -> MLForecast:
    model = MLForecast(
        models=[LGBMRegressor()],
        freq="B",
        lags=[1, 5, 10],
        lag_transforms={
            1: [(rolling_mean, 5), (rolling_max, 5), (rolling_min, 5)],
        },
        num_threads=2,
    )
    return model


def load_xgboost() -> MLForecast:
    model = MLForecast(
        models=[XGBRegressor()],
        freq="B",
        lags=[1, 5, 10],
        lag_transforms={
            1: [(rolling_mean, 5), (rolling_max, 5), (rolling_min, 5)],
        },
        num_threads=2,
    )
    return model


def fit_predict_boosting_model(model_name: str, mts: MultipleTimeSeries) -> dict:
    if "LGBMRegressor" in model_name:
        model = load_lightgbm()
    elif "XGBRegressor" in model_name:
        model = load_xgboost()
    else:
        raise Exception(f"Name '{model_name}' is not a valid booster model name.")
    model.fit(mts.get_train_df())
    results = model.predict(h=len(mts.y_test))
    y_preds = get_y_preds_from_boosting_results(results, mts.names, model_name)
    return y_preds


def validate_boosting_model(
    model_name: str,
    mts: MultipleTimeSeries,
    n_validations: int = model_config["n_validations"],
) -> tuple[float, float]:
    if "LGBMRegressor" in model_name:
        model = load_lightgbm()
    elif "XGBRegressor" in model_name:
        model = load_xgboost()
    else:
        raise Exception(f"Name '{model_name}' is not a valid booster model name.")
    test_days = len(mts.y_test)
    mae_lst, rmse_lst = [], []
    for _ in range(n_validations):
        trial_i = random.randint(0, len(mts.x_train) - 1 - test_days)
        y_true = mts.x_train[trial_i + 1 : trial_i + 1 + test_days, -1, :]
        model.fit(mts.get_train_df(trial_i))
        results = model.predict(h=test_days)
        y_preds = get_y_preds_from_boosting_results(results, mts.names, model_name)
        y_pred = np.stack(list(y_preds.values()), 1)
        assert y_true.shape == y_pred.shape == (test_days, len(mts.names))
        metrics = evaluate_return_predictions(
            mts.get_returns_from_features(y_true),
            mts.get_returns_from_features(y_pred),
        )
        mae_lst.append(metrics["MAE"])
        rmse_lst.append(metrics["RMSE"])
    return float(np.mean(mae_lst)), float(np.mean(rmse_lst))


def get_y_preds_from_boosting_results(
    results: pd.DataFrame, ts_names: list, model_name: str
) -> dict:
    return {
        ts_name: list(
            results[results["unique_id"] == ts_name][model_name.split("_")[-1]]
        )
        for ts_name in ts_names
    }
