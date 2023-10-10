import optuna
import pandas as pd

from config.config import model_config, paths
from models.base import (
    validate_arima,
    validate_exponential_smoothing,
    validate_moving_average,
    validate_prophet,
)
from models.boosting import validate_boosting_model
from models.lstms import validate_lstm_model
from utils.data_classes import MultipleTimeSeries
from utils.data_preprocessing import preprocess_data
from utils.file_handling import write_csv_results


def validate_model(model_name: str, mts: MultipleTimeSeries) -> tuple[float, float]:
    if model_name == "arima":
        mae, rmse = validate_arima(mts)
    elif model_name == "exponential_smoothing":
        mae, rmse = validate_exponential_smoothing(mts)
    elif model_name == "LGBMRegressor":
        mae, rmse = validate_boosting_model(model_name, mts)
    elif model_name == "moving_average":
        mae, rmse = validate_moving_average(mts, recursive=False)
    elif model_name == "moving_average_recursive":
        mae, rmse = validate_moving_average(mts, recursive=True)
    elif model_name == "prophet":
        mae, rmse = validate_prophet(mts)
    elif model_name == "XGBRegressor":
        mae, rmse = validate_boosting_model(model_name, mts)
    else:
        raise Exception(f"Name '{model_name}' is not a valid model name.")
    return mae, rmse


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


if __name__ == "__main__":
    hparam = "look_back_window_size"
    results = []

    for model_name in model_config["names"]:
        if model_name == "lstm":
            print(
                f"Skipping optimization of {hparam} for LSTM model as this would require retraining."
            )
            continue

        def objective(trial):
            lbws = trial.suggest_int(hparam, 5, 2600, step=5)
            mts = preprocess_data(
                paths["csv"],
                look_back_window_size=lbws,
                include_stock_index=True,
            )
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
    write_csv_results(pd.concat(results), "optimization_lstm")
