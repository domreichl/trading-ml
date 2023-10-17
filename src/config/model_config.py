import os

root_dir = os.path.join(os.path.dirname(__file__), "..", "..")

model_config = {
    "names": [
        "arima",
        "exponential_smoothing",
        "LGBMRegressor",
        "lstm",
        "moving_average",
        "moving_average_recursive",
        "prophet",
        "XGBRegressor",
    ],
    "n_validations": 50,
    "seasonal_periods": 20,
    "n_epochs": 10,
    "batch_size": 30,
    "dropout_rate": 0.3,
    "ckpt_dir": os.path.join(root_dir, "ckpts"),
    "results_dir": os.path.join(root_dir, "results"),
    "results_cols": ["Model", "Target", "Metric", "Score"],
}
