import pandas as pd

from config.model_config import model_config
from pipeline.train import train_model
from utils.data_preprocessing import preprocess_data
from utils.data_processing import get_forecast_df
from utils.file_handling import reset_dir, load_csv_results, write_csv_results
from utils.prediction import generate_predictions


if __name__ == "__main__":
    predictions = []
    reset_dir("prod")
    selected = load_csv_results("selection")["Model"].unique()
    for i, model_name in enumerate(model_config["names"]):
        if "eval_" + model_name not in selected:
            continue
        model_name = "prod_" + model_name
        print(f"Computing forecast with model '{model_name}'")
        deep_learning = False
        if "lstm" in model_name:
            deep_learning = True
        mts = preprocess_data()
        mts.merge_features(for_deep_learning=deep_learning)
        if i in model_config["trainable"]:
            train_model(model_name, mts)
        returns_predicted, prices_predicted = generate_predictions(
            model_name, mts, forecast=True
        )
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
