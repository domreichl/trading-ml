from models.boosting import validate_boosting_model
from models.local import (
    validate_arima,
    validate_exponential_smoothing,
    validate_moving_average,
    validate_prophet,
)
from models.neural_nets import RegressionNet
from utils.data_classes import MultipleTimeSeries


def validate_model(
    model_name: str, mts: MultipleTimeSeries, n_validations: int
) -> tuple[float, float]:
    if "arima" in model_name:
        rmse, ps, acc = validate_arima(mts, n_validations)
    elif "exponential_smoothing" in model_name:
        rmse, ps, acc = validate_exponential_smoothing(mts, n_validations)
    elif "LGBMRegressor" in model_name:
        rmse, ps, acc = validate_boosting_model(model_name, mts, n_validations)
    elif "_net" in model_name:
        model = RegressionNet(model_name, mts)
        model.load()
        rmse, ps, acc = model.validate(n_validations)
    elif "moving_average_recursive" in model_name:
        rmse, ps, acc = validate_moving_average(mts, n_validations, recursive=True)
    elif "prophet" in model_name:
        rmse, ps, acc = validate_prophet(mts, n_validations)
    elif "XGBRegressor" in model_name:
        rmse, ps, acc = validate_boosting_model(model_name, mts, n_validations)
    else:
        raise Exception(f"Name '{model_name}' is not a valid model name.")
    return rmse, ps, acc
