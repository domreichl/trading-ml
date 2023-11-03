import numpy as np
import pandas as pd
from typing import Optional

from utils.data_processing import compute_predicted_return, compute_predicted_returns
from utils.file_handling import ResultsHandler


def recommend_stock(
    current_prices: dict,
    position_type: str,
    optimize: str,
    top_models: Optional[list[str]] = None,
    buy_price: float = 1000,
) -> str:
    candidates = {}
    forecast = ResultsHandler().load_csv_results("forecast")
    for model_name in forecast["Model"].unique():
        if top_models:
            if model_name.replace("prod_", "main_") not in top_models:
                continue
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
    print(f" - Model Agreement: {model_agreement}")
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
        stocks[col] = {
            "ModelsFavoringStock": (~candidates[col].isna()).sum(),
            "MedianPredictedReturn": candidates[col].median(),
        }
    stocks = pd.DataFrame(stocks).transpose()
    if optimize == "risk":
        stocks = stocks[
            stocks["ModelsFavoringStock"] == stocks["ModelsFavoringStock"].max()
        ].copy()
    elif optimize == "reward":
        pass  # accept that model agreement may be low
    top_stock = get_most_lucrative_stock(stocks, position_type)
    ISIN = top_stock.index[0]
    predicted_return = round((top_stock["MedianPredictedReturn"].iloc[0]), 5)
    model_agreement = (
        f"{int(top_stock['ModelsFavoringStock'].iloc[0])}/{len(candidates)}"
    )
    return ISIN, predicted_return, model_agreement


def get_most_lucrative_stock(stocks: pd.DataFrame, position_type: str) -> pd.DataFrame:
    if position_type == "long":
        return stocks[
            stocks["MedianPredictedReturn"] == stocks["MedianPredictedReturn"].max()
        ]
    elif position_type == "short":
        return stocks[
            stocks["MedianPredictedReturn"] == stocks["MedianPredictedReturn"].min()
        ]


def recommend_close_position(
    forecast: pd.DataFrame, current_price: float, position_type: str
) -> bool:
    predicted_returns = []
    for model_name in forecast["Model"].unique():
        predicted_returns.append(
            compute_predicted_return(
                current_price, forecast[forecast["Model"] == model_name]
            )
        )
    predicted_returns = np.array(predicted_returns)
    n_negative_return = len(predicted_returns[predicted_returns < 1.0])
    print(
        f"\n{n_negative_return}/{len(predicted_returns)} top models predict a negative return."
    )
    if n_negative_return / len(predicted_returns) > 0.5:
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
