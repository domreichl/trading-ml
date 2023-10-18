import pandas as pd

from pipeline.select import pick_top_models
from utils.data_preprocessing import preprocess_data
from utils.file_handling import write_json_results
from utils.indicators import compute_market_signals, interpret_market_signals
from utils.recommendation import recommend_stock


if __name__ == "__main__":
    mts = preprocess_data()
    results = {"buy_price": 1000}
    current_prices = {ISIN: cp[-1] for ISIN, cp in mts.close_prices.items()}
    for position_type in ["short", "long"]:
        top_models = pick_top_models(position_type, prod=True)
        results[position_type] = {"Top Models": top_models}
        for optimize in ["risk", "return"]:
            top_stock, predicted_return, model_agreement = recommend_stock(
                top_models,
                current_prices,
                position_type,
                optimize,
                results["buy_price"],
            )
            trend, state = compute_market_signals(mts.close_prices[top_stock])
            trend, state = interpret_market_signals(top_stock, trend, state)
            results[position_type][optimize] = {
                "Top Stock": top_stock,
                "Predicted Return": predicted_return,
                "Model Agreement": model_agreement,
                "Market State": trend,
                "Market Trend": state,
            }
    write_json_results(results, "recommendation")
