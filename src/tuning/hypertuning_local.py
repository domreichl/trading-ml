import optuna

from utils.data_preprocessing import preprocess_data
from utils.file_handling import ResultsHandler
from utils.validation import validate_model


MODEL_NAME = "prophet"
N_VALIDATIONS = 100


def objective(trial):
    lbws = trial.suggest_int("look_back_window_size", 10, 1300, 5)
    mts = preprocess_data("exp.csv", look_back_window_size=lbws)
    rmse, ps, acc = validate_model(MODEL_NAME, mts, N_VALIDATIONS)
    return rmse, ps, acc


study = optuna.create_study(
    directions=["minimize", "maximize"],
    sampler=optuna.samplers.GridSampler(
        {
            "look_back_window_size": [5, 10, 15, 20, 22, 25, 65, 260, 520, 780, 1300],
        }
    ),
)
study.optimize(objective)

df = study.trials_dataframe()
df["Model"] = MODEL_NAME
df = df[["Model", "params_look_back_window_size", "values_0", "values_1", "values_2"]]
df.columns = ["Model", "LookBackWindowSize", "RMSE", "PredictiveScore", "Accuracy"]
ResultsHandler().write_csv_results(df, f"tuning/{MODEL_NAME}")
