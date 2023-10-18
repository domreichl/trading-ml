model_config = {
    "names": [
        "arima",
        "exponential_smoothing",
        "LGBMRegressor",
        "lstm",
        "moving_average_recursive",
        "prophet",
        "XGBRegressor",
    ],
    "trainable": [0, 3, 5],
    "look_back_window_size": 260,
    "test_days": 10,
    "seasonal_periods": 20,
    "n_validations": 50,
}
