import numpy as np

from utils.data_classes import MultipleTimeSeries


def generate_predictions(
    model_name: str, mts: MultipleTimeSeries, forecast: bool = False
) -> tuple[dict, dict]:
    if "arima" in model_name:
        from models.local import predict_arima

        y_preds = predict_arima(mts, model_name)
    elif "exponential_smoothing" in model_name:
        from models.local import fit_predict_exponential_smoothing

        y_preds = fit_predict_exponential_smoothing(mts)
    elif "LGBMRegressor" in model_name:
        from models.boosting import fit_predict_boosting_model

        y_preds = fit_predict_boosting_model(model_name, mts)
    elif "lstm" in model_name:
        from models.lstms import load_lstm_model

        model = load_lstm_model(model_name, mts)
        y_preds = model.predict()
    elif "moving_average_recursive" in model_name:
        from models.local import predict_moving_average_recursive

        y_preds = predict_moving_average_recursive(mts)
    elif "prophet" in model_name:
        from models.local import predict_prophet

        y_preds = predict_prophet(mts, model_name)
    elif "XGBRegressor" in model_name:
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
