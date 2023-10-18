import pandas as pd

from config.model_config import model_config
from models.base import (
    validate_arima,
    validate_exponential_smoothing,
    validate_moving_average,
    validate_prophet,
)
from models.boosting import validate_boosting_model
from models.lstms import load_lstm_model
from utils.data_classes import MultipleTimeSeries
from utils.data_preprocessing import preprocess_data
from utils.file_handling import write_csv_results


def validate_model(model_name: str, mts: MultipleTimeSeries) -> tuple[float, float]:
    if "arima" in model_name:
        mae, rmse = validate_arima(mts)
    elif "exponential_smoothing" in model_name:
        mae, rmse = validate_exponential_smoothing(mts)
    elif "LGBMRegressor" in model_name:
        mae, rmse = validate_boosting_model(model_name, mts)
    elif "lstm" in model_name:
        model = load_lstm_model(model_name, mts)
        mae, rmse = model.validate()
    elif "moving_average_recursive" in model_name:
        mae, rmse = validate_moving_average(mts, recursive=True)
    elif "prophet" in model_name:
        mae, rmse = validate_prophet(mts)
    elif "XGBRegressor" in model_name:
        mae, rmse = validate_boosting_model(model_name, mts)
    else:
        raise Exception(f"Name '{model_name}' is not a valid model name.")
    return mae, rmse


if __name__ == "__main__":
    mts = preprocess_data()
    models, maes, rmses = [], [], []
    for model_name in model_config["names"]:
        model_name = "eval_" + model_name
        print(
            f"Validating {model_name} with {model_config['n_validations']} iterations on train set..."
        )
        mae, rmse = validate_model(model_name, mts)
        models.append(model_name)
        maes.append(mae)
        rmses.append(rmse)
        print(f"Results for {model_name}: MAE={round(mae, 4)}, RMSE={round(rmse, 4)}")
    results = pd.DataFrame({"Model": models, "MAE": maes, "RMSE": rmses})
    write_csv_results(results, "validation")
