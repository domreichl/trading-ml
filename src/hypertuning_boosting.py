import optuna

from utils.data_preprocessing import preprocess_data
from utils.file_handling import ResultsHandler
from utils.validation import validate_model


MODEL_NAME = "LGBMRegressor"
N_VALIDATIONS = 50


def objective(trial):
    lbws = trial.suggest_int("look_back_window_size", 10, 1300, 5)
    mts = preprocess_data("exp.csv", look_back_window_size=lbws)
    mae, _, f1 = validate_model(MODEL_NAME, mts, N_VALIDATIONS)
    return mae, f1


study = optuna.create_study(
    directions=["minimize", "maximize"],
    sampler=optuna.samplers.GridSampler(
        {
            "look_back_window_size": [10, 65, 260, 520, 780, 1300],
        }
    ),
)
study.optimize(objective)

df = study.trials_dataframe()
df["Model"] = MODEL_NAME
df = df[["Model", "params_look_back_window_size", "values_0", "values_1"]]
df.columns = ["Model", "LookBackWindowSize", "MAE", "F1-Score"]
df.sort_values("F1-Score", inplace=True)

ResultsHandler().write_csv_results(df, f"tuning/{MODEL_NAME}")

param = {
    "boosting_type": "gbdt",
    "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
    "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    "num_leaves": trial.suggest_int("num_leaves", 2, 256),
    "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
    "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
}
