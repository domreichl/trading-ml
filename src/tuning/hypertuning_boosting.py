import optuna

from models.boosting import validate_boosting_model
from utils.data_preprocessing import preprocess_data
from utils.file_handling import ResultsHandler


MODEL_NAME = "LGBMRegressor"  # "XGBRegressor"
N_VALIDATIONS = 50


def objective(trial):
    lbws = trial.suggest_int("look_back_window_size", 260, 1300, 260)
    mts = preprocess_data("exp.csv", look_back_window_size=lbws)
    rmse, ps = validate_boosting_model(MODEL_NAME, mts, N_VALIDATIONS)
    return rmse, ps


study = optuna.create_study(
    directions=["minimize", "maximize"],
    sampler=optuna.samplers.GridSampler({"look_back_window_size": [260, 520, 780]}),
)
study.optimize(objective)

df = study.trials_dataframe()
df["Model"] = MODEL_NAME
df = df[
    [
        "Model",
        "params_look_back_window_size",
        "values_0",
        "values_1",
    ]
]
df.columns = ["Model", "LookBackWindowSize", "RMSE", "PredictiveScore"]
ResultsHandler().write_csv_results(df, f"tuning/{MODEL_NAME}")
