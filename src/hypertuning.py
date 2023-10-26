import optuna
import pandas as pd

from utils.data_preprocessing import preprocess_data
from utils.file_handling import ResultsHandler
from utils.validation import validate_model


MODEL_NAME = "moving_average_recursive"
N_VALIDATIONS = 100


def objective(trial):
    lbws = trial.suggest_int("look_back_window_size", 10, 1300, 5)
    normalization = trial.suggest_int("normalization", 0, 1)
    mts = preprocess_data(
        "exp.csv", look_back_window_size=lbws, normalize=bool(normalization)
    )
    mae, _, f1 = validate_model(MODEL_NAME, mts, N_VALIDATIONS)
    return mae, f1


study = optuna.create_study(
    directions=["minimize", "maximize"],
    sampler=optuna.samplers.GridSampler(
        {
            # "look_back_window_size": [10, 20, 65, 260, 520, 780, 1300],
            "look_back_window_size": [10, 15, 20, 25, 65, 260, 520, 780, 1040, 1300],
            "normalization": [0, 1],
        }
    ),
)
study.optimize(objective)

df = study.trials_dataframe()
df["Model"] = MODEL_NAME
df = df[
    [
        "Model",
        "params_look_back_window_size",
        "params_normalization",
        "values_0",
        "values_1",
    ]
]
df.columns = ["Model", "LookBackWindowSize", "Normalization", "MAE", "F1-Score"]
df.sort_values("F1-Score", inplace=True)
ResultsHandler().write_csv_results(df, f"tuning/{MODEL_NAME}")
