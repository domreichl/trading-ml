import pandas as pd

from utils.config import Config
from utils.data_preprocessing import preprocess_data
from utils.data_processing import get_df_from_predictions
from utils.evaluation import compute_prediction_performances
from utils.prediction import generate_predictions
from utils.file_handling import ResultsHandler


if __name__ == "__main__":
    mts = preprocess_data()
    cfg = Config()
    rh = ResultsHandler()
    returns_actual = mts.get_test_returns()
    prices_actual = mts.get_test_prices()
    dates = mts.get_test_dates()
    naive_errors = mts.get_naive_errors()
    performance, predictions = [], []
    for model_name in cfg.model_names:
        model_name = "eval_" + model_name
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
    rh.write_csv_results(performance, "test_performance")
    rh.write_csv_results(predictions, "test_predictions")
    rh.write_frontend_data(performance, "test_performance")
    rh.write_frontend_data(
        predictions.drop(columns=["Return", "ReturnPredicted"]), "test_predictions"
    )
