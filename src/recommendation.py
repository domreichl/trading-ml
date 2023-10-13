import os
import pandas as pd

from config.config import data_config, model_config, paths
from utils.data_preprocessing import preprocess_data


def recommend_stock(ratings: dict) -> str:
    # TODO: continue here
    return "someISIN"


def compute_ratings(current_prices: dict) -> dict:
    validation = pd.read_csv(os.path.join(paths["results"], "validation.csv"))
    performance = pd.read_csv(os.path.join(paths["results"], "performance.csv"))
    forecast = pd.read_csv(os.path.join(paths["results"], "forecast.csv"))
    ratings = {}
    for model_name in model_config["names"]:
        ratings[model_name] = {}
        rmse = float(validation[validation["Model"] == model_name]["RMSE"])
        perf = performance[performance["Model"] == model_name]
        f1 = float(perf[perf["Metric"] == "F1"]["Score"])
        smape = float(perf[perf["Metric"] == "SMAPE"]["Score"])
        fc = forecast[forecast["Model"] == "prod_" + model_name]
        for ISIN, current_price in current_prices.items():
            ratings[model_name][ISIN] = {}
            predicted_price = float(fc[fc["ISIN"] == ISIN]["Price"].iloc[-1])
            predicted_return = predicted_price / current_price * 100
            if predicted_return < 100:
                continue  # price predicted to decrease
            ratings[model_name][ISIN]["PredictedReturn"] = predicted_return
            ratings[model_name][ISIN]["Rating"] = predicted_return / rmse * f1 / smape
            print(ratings)


mts = preprocess_data(
    paths["csv"],
    look_back_window_size=data_config["look_back_window_size"],
    include_stock_index=True,
)
current_prices = {ISIN: cp[-1] for ISIN, cp in mts.close_prices.items()}
# TODO: fix out-of-bounds error
compute_ratings(current_prices)

"""
    rank ISINs by
        (a) number of models predictions profits
            with models weighted by metrics
                /RMSE
                *F1
            calculated from validation
            & previous week's performance
        (b) % of price increase expected
    for a single point metric, compute (a)*(b)
        but also analyze all factors separately
    print separate factors, recommendation formula, and final result
    invest in ISN with highest final result (rank/custom metric)
    stay invested until some models predict a price drop (compute new predictions every 1-5 days)
    use 2-6 technical indicators for further guidance
"""
