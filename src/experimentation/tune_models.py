import optuna, random
import numpy as np
import pandas as pd
from dvclive import Live
from dvclive.optuna import DVCLiveCallback
from sktime.forecasting.arima import AutoARIMA

from utils.data_preprocessing import preprocess_data
from utils.data_processing import get_signs_from_returns
from utils.evaluation import evaluate_return_predictions, evaluate_sign_predictions
from utils.file_handling import ResultsHandler
from utils.validation import validate_model


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


def get_grid_sampler(hparam: str):
    week = 5
    year = week * 52
    search_space = {
        hparam: [
            week * 4,
            year // 4,
            year,
            year * 2,
        ]
    }
    grid_sampler = optuna.samplers.GridSampler(search_space)
    return grid_sampler


hparam = "look_back_window_size"
results = []


def objective(trial):
    lbws = trial.suggest_int(hparam, 5, 2600, step=5)
    mts = preprocess_data(look_back_window_size=lbws)
    _, rmse = validate_model(model_name, mts)
    return rmse


study = optuna.create_study(
    direction="minimize",
    sampler=get_grid_sampler(hparam),
    study_name=f"{model_name}_{hparam}",
)
study.optimize(objective)
study_df = study.trials_dataframe()
study_df["Model"] = model_name
study_df = study_df[["Model", f"params_{hparam}", "value"]]
study_df.columns = ["Model", hparam, "RMSE"]
study_df.sort_values("RMSE", inplace=True)
study_df = study_df.reset_index(drop=True)
results.append(study_df)
ResultsHandler().write_csv_results(pd.concat(results), "optimization_lstm")
