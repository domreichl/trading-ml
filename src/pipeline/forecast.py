import pandas as pd

from pipeline.train import train_model
from utils.config import Config
from utils.data_preprocessing import preprocess_data
from utils.data_processing import get_forecast_df
from utils.file_handling import ResultsHandler, CkptHandler
from utils.prediction import generate_predictions


if __name__ == "__main__":
    predictions = []
    rh = ResultsHandler()
    CkptHandler().reset_dir("prod")
    cfg = Config()
    selected = rh.load_csv_results("selection")["Model"].unique()
    for model_name, model_cfg in cfg.models.items():
        if "eval_" + model_name not in selected:
            continue
        model_name = "prod_" + model_name
        print(f"Computing forecast with model '{model_name}'")
        deep_learning = False
        if "lstm" in model_name:
            deep_learning = True
        mts = preprocess_data()
        mts.merge_features(for_deep_learning=deep_learning)
        if model_cfg["store_ckpt"]:
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
    rh.write_csv_results(predictions, "forecast")
