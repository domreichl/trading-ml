import os, random
import numpy as np
import pandas as pd

from config.config import data_config, paths
from utils.data_preprocessing import preprocess_data
from utils.data_processing import compute_predicted_return
from utils.indicators import compute_market_signals, print_market_signals
from utils.model_selection import pick_top_models


def recommend_close_position(
    ISIN: str, current_price: float, position_type: str
) -> bool:
    forecast = pd.read_csv(os.path.join(paths["results"], "forecast.csv"))
    forecast = forecast[forecast["ISIN"] == ISIN]
    predicted_returns = []
    for model_name in pick_top_models(position_type):
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


if __name__ == "__main__":
    mts = preprocess_data(
        paths["csv"],
        look_back_window_size=data_config["look_back_window_size"],
        include_stock_index=True,
    )
    ISIN = random.choice(list(mts.close_prices.keys()))
    current_price = mts.close_prices[ISIN][-1]
    for position_type in ["short", "long"]:
        recommend_close_position(ISIN, current_price, position_type)
    overbought, bullish = compute_market_signals(mts.close_prices[ISIN])
    print_market_signals(ISIN, overbought, bullish)
