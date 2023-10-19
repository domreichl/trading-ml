import random
import numpy as np
from pmdarima import auto_arima
from prophet import Prophet
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.compose import EnsembleForecaster

from utils.data_classes import MultipleTimeSeries
from utils.evaluation import evaluate_return_predictions
from utils.file_handling import CkptHandler


def fit_arima(mts: MultipleTimeSeries, model_name: str) -> None:
    for i, ts_name in enumerate(mts.names):
        model = auto_arima(mts.x_train[-1, :, i])
        CkptHandler().save_model_to_pickle_ckpt(model, f"{model_name}_{ts_name}")


def predict_arima(mts: MultipleTimeSeries, model_name: str) -> dict:
    y_preds = {}
    for ts_name in mts.names:
        model = CkptHandler().load_model_from_pickle_ckpt(f"{model_name}_{ts_name}")
        y_preds[ts_name] = model.predict(len(mts.y_test))
    return y_preds


def validate_arima(mts: MultipleTimeSeries, n_validations: int) -> tuple[float, float]:
    test_days = len(mts.y_test)
    mae_lst, rmse_lst = [], []
    for i, ts_name in enumerate(mts.names):
        print(f"Validating arima_{ts_name}")
        for _ in range(n_validations):
            trial_i = random.randint(0, len(mts.x_train) - 1 - test_days)
            model = auto_arima(mts.x_train[trial_i, :, i])
            y_true = mts.x_train[trial_i + 1 : trial_i + 1 + test_days, -1, i]
            y_pred = model.predict(test_days)
            assert y_true.shape == y_pred.shape == (test_days,)
            metrics = evaluate_return_predictions(
                mts.get_returns_from_features(y_true),
                mts.get_returns_from_features(y_pred),
            )
            mae_lst.append(metrics["MAE"])
            rmse_lst.append(metrics["RMSE"])
    return float(np.mean(mae_lst)), float(np.mean(rmse_lst))


def fit_predict_exponential_smoothing(mts: MultipleTimeSeries) -> dict:
    model = load_exponential_smoothing()
    results = model.fit_predict(mts.x_train[-1, :, :], fh=range(1, len(mts.y_test) + 1))
    y_preds = {name: results[:, i] for i, name in enumerate(mts.names)}
    return y_preds


def validate_exponential_smoothing(
    mts: MultipleTimeSeries, n_validations: int
) -> tuple[float, float]:
    test_days = len(mts.y_test)
    model = load_exponential_smoothing()
    mae_lst, rmse_lst = [], []
    for _ in range(n_validations):
        trial_i = random.randint(0, len(mts.x_train) - 1 - test_days)
        y_true = mts.x_train[trial_i + 1 : trial_i + 1 + test_days, -1, :]
        y_pred = model.fit_predict(
            mts.x_train[trial_i, :, :], fh=range(1, test_days + 1)
        )
        assert y_true.shape == y_pred.shape == (test_days, len(mts.names))
        metrics = evaluate_return_predictions(
            mts.get_returns_from_features(y_true),
            mts.get_returns_from_features(y_pred),
        )
        mae_lst.append(metrics["MAE"])
        rmse_lst.append(metrics["RMSE"])
    return float(np.mean(mae_lst)), float(np.mean(rmse_lst))


def load_exponential_smoothing(seasonal_periods: int = 20) -> EnsembleForecaster:
    model = EnsembleForecaster(
        [
            ("ses", ExponentialSmoothing(sp=seasonal_periods)),
            (
                "holt",
                ExponentialSmoothing(
                    trend="add",
                    damped_trend=False,
                    sp=seasonal_periods,
                ),
            ),
            (
                "damped",
                ExponentialSmoothing(
                    trend="add",
                    damped_trend=True,
                    sp=seasonal_periods,
                ),
            ),
        ]
    )
    return model


def predict_moving_average(mts: MultipleTimeSeries, idx: int = -1) -> dict:
    test_days = len(mts.y_test)
    if idx == -1:
        fh = mts.y_test
    else:
        fh = np.concatenate([mts.y_train[:, -1, :], mts.y_test], 0)[
            idx : idx + test_days
        ]
    features = np.concatenate([mts.x_train[idx, :, :], fh], 0)
    y_preds = {}
    for i, name in enumerate(mts.names):
        y_preds[name] = [
            np.mean(features[j : j - test_days, i]) for j in range(test_days)
        ]
    return y_preds


def predict_moving_average_recursive(mts: MultipleTimeSeries, idx: int = -1) -> dict:
    test_days = len(mts.y_test)
    features = mts.x_train[idx, :, :]
    y_preds = {}
    for i, name in enumerate(mts.names):
        x = features[:, i]
        for _ in range(test_days):
            pred = np.expand_dims(np.mean(x), 0)
            x = np.concatenate([x[1:], pred], 0)
        y_preds[name] = x[-test_days:]
    return y_preds


def validate_moving_average(
    mts: MultipleTimeSeries, n_validations: int, recursive: bool
) -> tuple[float, float]:
    test_days = len(mts.y_test)
    mae_lst, rmse_lst = [], []
    for _ in range(n_validations):
        trial_i = random.randint(0, len(mts.x_train) - 1 - test_days)
        if recursive:
            y_preds = predict_moving_average_recursive(mts, trial_i)
        else:
            y_preds = predict_moving_average(mts, trial_i)
        y_pred = np.stack(list(y_preds.values()), 1)
        y_true = mts.x_train[trial_i + 1 : trial_i + 1 + test_days, -1, :]
        assert y_true.shape == y_pred.shape == (test_days, len(mts.names))
        metrics = evaluate_return_predictions(
            mts.get_returns_from_features(y_true),
            mts.get_returns_from_features(y_pred),
        )
        mae_lst.append(metrics["MAE"])
        rmse_lst.append(metrics["RMSE"])
    return float(np.mean(mae_lst)), float(np.mean(rmse_lst))


def fit_prophet(mts: MultipleTimeSeries, model_name: str) -> None:
    df = mts.get_train_df()
    for ts_name in mts.names:
        x = df[df["unique_id"] == ts_name].copy()
        model = Prophet().fit(x)
        CkptHandler().save_model_to_json_ckpt(model, f"{model_name}_{ts_name}")


def predict_prophet(mts: MultipleTimeSeries, model_name: str) -> dict:
    y_preds = {}
    for ts_name in mts.names:
        model = CkptHandler().load_model_from_json_ckpt(f"{model_name}_{ts_name}")
        results = model.predict(
            model.make_future_dataframe(periods=len(mts.y_test), include_history=False)
        )
        y_preds[ts_name] = list(results["yhat"])
    return y_preds


def validate_prophet(mts: MultipleTimeSeries, n_validations: int) -> dict:
    test_days = len(mts.y_test)
    mae_lst, rmse_lst = [], []
    for i, ts_name in enumerate(mts.names):
        print(f"Validating prophet_{ts_name}")
        for _ in range(n_validations):
            trial_i = random.randint(0, len(mts.x_train) - 1 - test_days)
            model = Prophet().fit(mts.get_train_df(trial_i, ts_name))
            y_true = mts.x_train[trial_i + 1 : trial_i + 1 + test_days, -1, i]
            y_pred = np.array(
                model.predict(
                    model.make_future_dataframe(
                        periods=len(mts.y_test), include_history=False
                    )
                )["yhat"]
            )
            assert y_true.shape == y_pred.shape == (test_days,)
            metrics = evaluate_return_predictions(
                mts.get_returns_from_features(y_true),
                mts.get_returns_from_features(y_pred),
            )
            mae_lst.append(metrics["MAE"])
            rmse_lst.append(metrics["RMSE"])
    return float(np.mean(mae_lst)), float(np.mean(rmse_lst))
