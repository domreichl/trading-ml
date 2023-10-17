import optuna
import pandas as pd

from pipeline.validate import validate_model
from config.model_config import model_config
from utils.data_preprocessing import preprocess_data
from utils.file_handling import write_csv_results


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
        if "lstm" in model_name:
            print(
                f"Skipping optimization of {hparam} for LSTM model '{model_name}' as this would require retraining."
            )
            continue

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
    write_csv_results(pd.concat(results), "optimization_lstm")
