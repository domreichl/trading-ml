import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from xgboost import XGBRegressor

from utils.data_classes import MultipleTimeSeries
from utils.evaluation import get_validation_metrics


def load_lightgbm() -> MLForecast:
    model = MLForecast(
        models=[
            LGBMRegressor(boosting_type="gbdt", n_estimators=10, learning_rate=0.15)
        ],
        freq="B",
        lags=[1, 5],
        num_threads=2,
    )
    return model


def load_xgboost() -> MLForecast:
    model = MLForecast(
        models=[XGBRegressor(booster="dart", n_estimators=10, learning_rate=0.15)],
        freq="B",
        lags=[1, 5],
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
    model_name: str, mts: MultipleTimeSeries, n_validations: int
) -> tuple[float, float]:
    if "LGBMRegressor" in model_name:
        model = load_lightgbm()
    elif "XGBRegressor" in model_name:
        model = load_xgboost()
    else:
        raise Exception(f"Name '{model_name}' is not a valid booster model name.")
    test_days = len(mts.y_test)
    rmse_lst, ps_lst = [], []
    for _ in range(n_validations):
        trial_i = random.randint(0, len(mts.x_train) - 1 - test_days)
        y_true = mts.x_train[trial_i + 1 : trial_i + 1 + test_days, -1, :]
        model.fit(mts.get_train_df(trial_i))
        results = model.predict(h=test_days)
        y_preds = get_y_preds_from_boosting_results(results, mts.names, model_name)
        y_pred = np.stack(list(y_preds.values()), 1)
        rmse, ps = get_validation_metrics(
            mts.get_returns_from_features(y_true), mts.get_returns_from_features(y_pred)
        )
        rmse_lst.append(rmse)
        ps_lst.append(ps)
    return float(np.mean(rmse_lst)), float(np.mean(ps_lst))


def get_y_preds_from_boosting_results(
    results: pd.DataFrame, ts_names: list, model_name: str
) -> dict:
    return {
        ts_name: list(
            results[results["unique_id"] == ts_name][model_name.split("_")[-1]]
        )
        for ts_name in ts_names
    }
