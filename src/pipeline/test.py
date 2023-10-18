import pandas as pd

from config.model_config import model_config
from utils.data_preprocessing import preprocess_data
from utils.data_processing import get_df_from_predictions
from utils.evaluation import compute_prediction_performances
from utils.prediction import generate_predictions
from utils.file_handling import write_csv_results, write_frontend_data


if __name__ == "__main__":
    mts = preprocess_data()
    returns_actual = mts.get_test_returns()
    prices_actual = mts.get_test_prices()
    dates = mts.get_test_dates()
    naive_errors = mts.get_naive_errors()
    performance, predictions = [], []
    for model_name in model_config["names"]:
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
    write_csv_results(performance, "test_performance")
    write_csv_results(predictions, "test_predictions")
    write_frontend_data(performance, "test_performance")
    write_frontend_data(
        predictions.drop(columns=["Return", "ReturnPredicted"]), "test_predictions"
    )
