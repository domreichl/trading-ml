import numpy as np
import pandas as pd

from config.model_config import model_config
from utils.data_classes import MultipleTimeSeries
from utils.data_preprocessing import preprocess_data
from utils.data_processing import get_df_from_predictions
from utils.evaluation import compute_prediction_performances
from utils.file_handling import write_csv_results, write_frontend_data


def generate_predictions(
    model_name: str, mts: MultipleTimeSeries, forecast: bool = False
) -> tuple[dict, dict]:
    if model_name in ["arima", "prod_arima"]:
        from models.base import fit_predict_arima

        y_preds = fit_predict_arima(mts, ckpt_dir=model_name)
    elif model_name in ["exponential_smoothing", "prod_exponential_smoothing"]:
        from models.base import fit_predict_exponential_smoothing

        y_preds = fit_predict_exponential_smoothing(mts)
    elif model_name in ["LGBMRegressor", "prod_LGBMRegressor"]:
        from models.boosting import fit_predict_boosting_model

        y_preds = fit_predict_boosting_model(model_name, mts)
    elif model_name in ["lstm", "prod_lstm"]:
        from models.lstms import get_lstm_model

        model = get_lstm_model(model_name, mts)
        y_preds = model.predict()
    elif model_name in ["moving_average"]:
        from models.base import predict_moving_average

        y_preds = predict_moving_average(mts)
    elif model_name in ["moving_average_recursive", "prod_moving_average_recursive"]:
        from models.base import predict_moving_average_recursive

        y_preds = predict_moving_average_recursive(mts)
    elif model_name in ["prophet", "prod_prophet"]:
        from models.base import fit_predict_prophet

        y_preds = fit_predict_prophet(mts, ckpt_dir=model_name)
    elif model_name in ["XGBRegressor", "prod_XGBRegressor"]:
        from models.boosting import fit_predict_boosting_model

        y_preds = fit_predict_boosting_model(model_name, mts)
    else:
        raise Exception(f"Name '{model_name}' is not a valid model name.")
    returns_predicted, prices_predicted = {}, {}
    for name, pred in y_preds.items():
        prices_predicted[name] = []
        if forecast:
            price = mts.close_prices[name][-1]
        else:
            price = mts.close_prices[name][-1 - len(mts.y_test)]
        returns_predicted[name] = list(mts.get_returns_from_features(np.array(pred)))
        for r in returns_predicted[name]:
            price *= r
            prices_predicted[name].append(round(price, 2))
    return returns_predicted, prices_predicted


if __name__ == "__main__":
    mts = preprocess_data()
    returns_actual = mts.get_test_returns()
    prices_actual = mts.get_test_prices()
    dates = mts.get_test_dates()
    naive_errors = mts.get_naive_errors()
    performance, predictions = [], []
    for model_name in model_config["names"]:
        print(f"Computing predictions with model '{model_name}'")
        returns_predicted, prices_predicted = generate_predictions(model_name, mts)
        performance.append(
            compute_prediction_performances(
                returns_actual,
                returns_predicted,
                prices_actual,
                prices_predicted,
                naive_errors,
                model_name,
            )
        )
        predictions.append(
            get_df_from_predictions(
                returns_actual,
                returns_predicted,
                prices_actual,
                prices_predicted,
                dates,
                model_name,
            )
        )
    performance = pd.concat(performance)
    predictions = pd.concat(predictions)
    write_csv_results(performance, "performance")
    write_csv_results(predictions, "predictions")
    write_frontend_data(
        predictions.drop(columns=["Return", "ReturnPredicted"]), "predictions"
    )
    write_frontend_data(performance, "performance")
