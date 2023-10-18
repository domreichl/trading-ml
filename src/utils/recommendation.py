import numpy as np
import pandas as pd

from pipeline.select import pick_top_models
from utils.data_processing import compute_predicted_return, compute_predicted_returns
from utils.file_handling import load_csv_results


def recommend_stock(
    top_models: list[str],
    current_prices: dict,
    position_type: str = "long",
    optimize: str = "return",
    buy_price: float = 1000,
) -> str:
    candidates = {}
    forecast = load_csv_results("forecast")
    for model_name in top_models:
        predicted_returns = compute_predicted_returns(
            current_prices,
            forecast[forecast["Model"] == model_name],
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
    return top_stock, predicted_return, model_agreement


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


def recommend_close_position(
    ISIN: str, current_price: float, position_type: str
) -> bool:
    forecast = load_csv_results("forecast")
    forecast = forecast[forecast["ISIN"] == ISIN]
    predicted_returns = []
    for model_name in pick_top_models(position_type, prod=True):
        predicted_returns.append(
            compute_predicted_return(
                current_price, forecast[forecast["Model"] == "prod_" + model_name]
            )
        )
    predicted_returns = np.array(predicted_returns)
    negative_return = len(predicted_returns[predicted_returns < 1.0])
    print(
        f"\n{negative_return}/{len(predicted_returns)} top models predict a negative return."
    )
    if negative_return / len(predicted_returns) > 0.5:
        if position_type == "long":
            recommendation = close()
        elif position_type == "short":
            recommendation = hold()
    else:
        if position_type == "long":
            recommendation = hold()
        elif position_type == "short":
            recommendation = close()
    return recommendation


def close() -> bool:
    print("You should therefore close the position.")
    return True


def hold() -> bool:
    print("You should therefore hold the position.")
    return False
