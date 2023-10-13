import os, json
import datetime as dt


config_dir = os.path.dirname(__file__)
root_dir = os.path.join(config_dir, "..", "..")

data_config = {
    "data_source": "wiener_boerse",  # 'wiener_boerse' (wienerboerse.at) or 'yfinance' (finance.yahoo.com)
    "start_date": dt.datetime(2000, 1, 3),  # Monday
    "end_date": dt.datetime(2023, 9, 22),  # Friday
    "stock_index": "AT0000999982",
    "securities": json.load(open(os.path.join(config_dir, "securities.json"))),
    "look_back_window_size": 260,  # 52 weeks x 5 weekdays
    "test_days": 10,
}

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
}

results_config = {
    "columns": ["Model", "Target", "Metric", "Score"],
}

paths = {
    "csv": os.path.join(
        root_dir,
        "data",
        f"{data_config['data_source']}_{len(data_config['securities'])}ts_{dt.datetime.strftime(data_config['start_date'], '%Y%m%d')}-{dt.datetime.strftime(data_config['end_date'], '%Y%m%d')}.csv",
    ),
    "ckpts": os.path.join(root_dir, "ckpts"),
    "results": os.path.join(root_dir, "results"),
    "frontend_data": os.path.join(root_dir, "frontend", "data"),
}
