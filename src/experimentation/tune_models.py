import random
import numpy as np
from dvclive import Live
from sktime.forecasting.arima import AutoARIMA

from utils.data_preprocessing import preprocess_data
from utils.data_processing import get_signs_from_returns
from utils.evaluation import evaluate_return_predictions, evaluate_sign_predictions


def validate_arima(mts, n_validations, sp) -> tuple[float, float]:
    test_days = len(mts.y_test)
    mae_lst, rmse_lst, f1_lst = [], [], []
    for i, ts_name in enumerate(mts.names):
        print(f"Validating arima_{ts_name}")
        for _ in range(n_validations):
            trial_i = random.randint(0, len(mts.x_train) - 1 - test_days)
            model = AutoARIMA(sp=sp, suppress_warnings=True)
            y_true = mts.x_train[trial_i + 1 : trial_i + 1 + test_days, -1, i]
            y_pred = np.squeeze(
                model.fit_predict(
                    mts.x_train[trial_i, :, i], fh=range(1, test_days + 1)
                )
            )
            assert y_true.shape == y_pred.shape == (test_days,)
            gt = mts.get_returns_from_features(y_true)
            pr = mts.get_returns_from_features(y_pred)
            metrics = evaluate_return_predictions(gt, pr)
            metrics_sign = evaluate_sign_predictions(
                get_signs_from_returns(gt), get_signs_from_returns(pr)
            )
            mae_lst.append(metrics["MAE"])
            rmse_lst.append(metrics["RMSE"])
            f1_lst.append(metrics_sign["F1"])
    return float(np.mean(mae_lst)), float(np.mean(rmse_lst)), float(np.mean(f1_lst))


mts = preprocess_data("exp.csv", look_back_window_size=780)

with Live() as live:
    for sp in [5, 1]:
        live.log_param("sp", sp)
        mae, rmse, f1 = validate_arima(mts, 10, sp)
        live.log_metric("MAE", mae)
        live.log_metric("RMSE", rmse)
        live.log_metric("F1", f1)
        live.next_step()
