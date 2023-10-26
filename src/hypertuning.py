import optuna

from utils.data_preprocessing import preprocess_data
from utils.file_handling import ResultsHandler
from utils.validation import validate_model


MODEL_NAME = "exponential_smoothing"
N_VALIDATIONS = 50


def objective(trial):
    lbws = trial.suggest_int("look_back_window_size", 10, 1300, 5)
    mts = preprocess_data("exp.csv", look_back_window_size=lbws)
    mae, _, f1 = validate_model(MODEL_NAME, mts, N_VALIDATIONS)
    return mae, f1


study = optuna.create_study(
    directions=["minimize", "maximize"],
    sampler=optuna.samplers.GridSampler(
        {"look_back_window_size": [10, 20, 65, 260, 520, 780, 1300]}
    ),
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
df.columns = ["Model", "LookBackWindowSize", "MAE", "F1-Score"]
df["DailySeasonality"] = "false"
df["WeeklySeasonality"] = "true"
df["YearlySeasonality"] = "false"
df.sort_values("F1-Score", inplace=True)
ResultsHandler().write_csv_results(df, f"tuning/{MODEL_NAME}")
