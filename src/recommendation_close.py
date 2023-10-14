import os
import pandas as pd

from config.config import data_config, paths
from utils.data_preprocessing import preprocess_data
from utils.model_selection import pick_top_models


def recommend_close_position(
    top_models: list[str],
    current_prices: dict,
    position_type: str = "long",
) -> str:
    # TODO
    return


if __name__ == "__main__":
    mts = preprocess_data(
        paths["csv"],
        look_back_window_size=data_config["look_back_window_size"],
        include_stock_index=True,
    )
    current_prices = {ISIN: cp[-1] for ISIN, cp in mts.close_prices.items()}
    for position_type in ["short", "long"]:
        top_models = pick_top_models(position_type)
        recommend_close_position(top_models, current_prices, position_type)
