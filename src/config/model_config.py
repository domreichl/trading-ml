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
    "results_cols": ["Model", "Target", "Metric", "Score"],
    "n_validations": 50,
    "seasonal_periods": 20,
    "n_epochs": 10,
    "batch_size": 30,
    "dropout_rate": 0.3,
}
