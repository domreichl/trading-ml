import pandas as pd

from utils.data_preprocessing import preprocess_data
from utils.data_processing import get_df_from_predictions, convert_metrics_df_to_dict
from utils.evaluation import (
    compute_prediction_performances,
    rate_models,
    filter_overfit_models,
)
from utils.prediction import generate_predictions
from utils.file_handling import ResultsHandler


rh = ResultsHandler()
top_val_models = rh.load_csv_results("validation_results")["Model"].unique()

performance, predictions = [], []
for model_name in top_val_models:
    mts = preprocess_data("main.csv", model_name=model_name.replace("val_", ""))
    returns_actual = mts.get_test_returns()
    prices_actual = mts.get_test_prices()
    dates = mts.get_test_dates()
    naive_error = mts.get_naive_error()
    model_name = model_name.replace("val_", "main_")
    print(f"Computing predictions with model '{model_name}'")
    returns_predicted, prices_predicted = generate_predictions(model_name, mts)
    performance.append(
        compute_prediction_performances(
            returns_actual,
            returns_predicted,
            prices_actual,
            prices_predicted,
            naive_error,
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
rh.write_frontend_data(performance, "test_metrics")
rh.write_frontend_data(
    predictions.drop(columns=["Return", "ReturnPredicted"]), "test_predictions"
)

model_ratings = rate_models(performance)
well_fit_models = filter_overfit_models(list(model_ratings.keys()))
rh.write_json_results(
    {k: v for k, v in model_ratings.items() if k in well_fit_models}, "test_ratings"
)
