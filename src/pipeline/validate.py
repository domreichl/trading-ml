import pandas as pd

from models.base import (
    validate_arima,
    validate_exponential_smoothing,
    validate_moving_average,
    validate_prophet,
)
from models.boosting import validate_boosting_model
from models.lstms import load_lstm_model
from utils.config import Config
from utils.data_classes import MultipleTimeSeries
from utils.data_preprocessing import preprocess_data
from utils.file_handling import ResultsHandler


def validate_model(
    model_name: str, mts: MultipleTimeSeries, n_validations: int
) -> tuple[float, float]:
    if "arima" in model_name:
        mae, rmse = validate_arima(mts, n_validations)
    elif "exponential_smoothing" in model_name:
        mae, rmse = validate_exponential_smoothing(mts, n_validations)
    elif "LGBMRegressor" in model_name:
        mae, rmse = validate_boosting_model(model_name, mts, n_validations)
    elif "lstm" in model_name:
        model = load_lstm_model(model_name, mts, n_validations)
        mae, rmse = model.validate()
    elif "moving_average_recursive" in model_name:
        mae, rmse = validate_moving_average(mts, n_validations, recursive=True)
    elif "prophet" in model_name:
        mae, rmse = validate_prophet(mts, n_validations)
    elif "XGBRegressor" in model_name:
        mae, rmse = validate_boosting_model(model_name, mts, n_validations)
    else:
        raise Exception(f"Name '{model_name}' is not a valid model name.")
    return mae, rmse


if __name__ == "__main__":
    mts = preprocess_data()
    cfg = Config()
    models, maes, rmses = [], [], []
    for model_name in cfg.model_names:
        model_name = "eval_" + model_name
        print(
            f"Validating {model_name} with {cfg.n_validations} iterations on train set..."
        )
        mae, rmse = validate_model(model_name, mts, cfg.n_validations)
        models.append(model_name)
        maes.append(mae)
        rmses.append(rmse)
        print(f"Results for {model_name}: MAE={round(mae, 4)}, RMSE={round(rmse, 4)}")
    results = pd.DataFrame({"Model": models, "MAE": maes, "RMSE": rmses})
    ResultsHandler().write_csv_results(results, "validation")
