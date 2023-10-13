import numpy as np
import pandas as pd

from prediction import generate_predictions
from config.config import data_config, model_config, paths
from utils.data_classes import MultipleTimeSeries
from utils.data_preprocessing import preprocess_data
from utils.data_processing import get_forecast_df
from utils.file_handling import write_csv_results, write_frontend_data


if __name__ == "__main__":
    mts = preprocess_data(
        paths["csv"],
        look_back_window_size=data_config["look_back_window_size"],
        include_stock_index=True,
    )
    mts.merge_features()
    predictions = []
    for model_name in model_config["names"]:
        if model_name == "moving_average":
            continue  # only moving_average_recursive possible
        deep_learning = False
        if model_name == "lstm":
            deep_learning = True
        mts = preprocess_data(
            paths["csv"],
            look_back_window_size=data_config["look_back_window_size"],
            include_stock_index=True,
        )
        mts.merge_features(for_deep_learning=deep_learning)
        model_name = "prod_" + model_name
        print(f"Computing forecast with model '{model_name}'")
        returns_predicted, prices_predicted = generate_predictions(model_name, mts)
        predictions.append(
            get_forecast_df(
                returns_predicted,
                prices_predicted,
                mts.get_forecast_dates(),
                model_name,
            )
        )
    predictions = pd.concat(predictions)
    write_csv_results(predictions, "forecast")
    write_frontend_data(predictions.drop(columns=["Return"]), "forecast")
