import pandas as pd

from utils.config import Config
from utils.data_preprocessing import preprocess_data
from utils.data_processing import get_df_from_predictions, convert_metrics_df_to_dict
from utils.evaluation import compute_prediction_performances, rank_models
from utils.prediction import generate_predictions
from utils.file_handling import ResultsHandler


rh = ResultsHandler()
performance, predictions = [], []
mts = preprocess_data("main.csv")
returns_actual = mts.get_test_returns()
prices_actual = mts.get_test_prices()
dates = mts.get_test_dates()
naive_errors = mts.get_naive_errors()
ranked_models = rh.load_csv_results("validation_ranked")["Model"].unique()

for model_name in ranked_models:
    model_name = model_name.replace("val_", "main_")
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
rh.write_csv_results(performance, "test_metrics")
rh.write_csv_results(predictions, "test_predictions")
rh.write_json_results(convert_metrics_df_to_dict(performance), "test_metrics")
rh.write_csv_results(rank_models(), "test_ranked")
rh.write_frontend_data(performance, "test_metrics")
rh.write_frontend_data(
    predictions.drop(columns=["Return", "ReturnPredicted"]), "test_predictions"
)
