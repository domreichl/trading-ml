import os
import pandas as pd

from config.config import data_config, paths
from utils.data_preprocessing import preprocess_data
from utils.data_processing import compute_predicted_returns
from utils.model_selection import pick_top_models


def recommend_stock(
    top_models: list[str],
    current_prices: dict,
    position_type: str = "long",
    optimize: str = "return",
    buy_price: float = 1000,
) -> str:
    candidates = {}
    forecast = pd.read_csv(os.path.join(paths["results"], "forecast.csv"))
    for model_name in top_models:
        predicted_returns = compute_predicted_returns(
            current_prices, forecast[forecast["Model"] == "prod_" + model_name]
        )
        candidates[model_name] = pick_top_stock_candidates(
            predicted_returns, position_type
        )
    top_stock, predicted_return, model_agreement = pick_top_stock(
        candidates, position_type, optimize
    )
    print(f"\nTop stock for a {position_type} position, optimized for {optimize}:")
    print(f" - ISIN: {top_stock}")
    print(f" - Model Agreement {model_agreement}%")
    print(f" - Predicted return: {predicted_return}")
    if position_type == "long":
        print(
            f" - Predicted gross profit when trading {buy_price}€: {round(buy_price*predicted_return-buy_price, 2)}€"
        )
    return top_stock


def pick_top_stock_candidates(
    predicted_returns: dict, position_type: str, n: int = 3
) -> list[str]:
    candidates = dict(
        list(
            dict(
                sorted(
                    predicted_returns.items(),
                    key=lambda item: item[1],
                    reverse=position_type == "long",
                )
            ).items()
        )[:n]
    )
    return candidates


def pick_top_stock(candidates: dict, position_type: str, optimize: str) -> tuple:
    stocks = {}
    candidates = pd.DataFrame(candidates).transpose()
    for col in candidates.columns:
        stocks[col] = {}
        stocks[col]["NaNs"] = candidates[col].isna().sum()
        stocks[col]["MeanPredictedReturn"] = candidates[col].mean()
    stocks = pd.DataFrame(stocks).transpose()
    if optimize == "risk":
        top_stocks = stocks[stocks["NaNs"] == stocks["NaNs"].min()]
    elif optimize == "return":
        top_stocks = stocks.copy()
    if position_type == "long":
        top_stock = top_stocks[
            top_stocks["MeanPredictedReturn"] == top_stocks["MeanPredictedReturn"].max()
        ].copy()
    elif position_type == "short":
        top_stock = top_stocks[
            top_stocks["MeanPredictedReturn"] == top_stocks["MeanPredictedReturn"].min()
        ].copy()
    top_stock["ModelAgreement"] = (
        (len(candidates) - top_stock["NaNs"]) / len(candidates) * 100
    )
    ISIN = top_stock.index[0]
    predicted_return = round((top_stock["MeanPredictedReturn"].iloc[0]), 5)
    model_agreement = round((top_stock["ModelAgreement"].iloc[0]))
    return ISIN, predicted_return, model_agreement


if __name__ == "__main__":
    mts = preprocess_data(
        paths["csv"],
        look_back_window_size=data_config["look_back_window_size"],
        include_stock_index=True,
    )
    current_prices = {ISIN: cp[-1] for ISIN, cp in mts.close_prices.items()}
    for position_type in ["short", "long"]:
        top_models = pick_top_models(position_type)
        for optimize in ["risk", "return"]:
            recommend_stock(top_models, current_prices, position_type, optimize)
