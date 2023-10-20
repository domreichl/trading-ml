import pandas as pd

from utils.data_preprocessing import preprocess_data
from utils.data_processing import get_forecast_df
from utils.file_handling import ResultsHandler, CkptHandler
from utils.prediction import generate_predictions
from utils.training import train_model


predictions = []
rh = ResultsHandler()
ranked_models = rh.load_csv_results("test_ranked")["Model"].unique()
CkptHandler().reset_dir("prod")

for model_name in ranked_models:
    model_name = model_name.replace("main_", "prod_")
    print(f"Computing forecast with model '{model_name}'")
    deep_learning = False
    if "lstm" in model_name:
        deep_learning = True
    mts = preprocess_data("main.csv")
    mts.merge_features(for_deep_learning=deep_learning)
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

rh.write_csv_results(pd.concat(predictions), "forecast")
