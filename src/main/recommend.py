from utils.data_preprocessing import preprocess_data
from utils.evaluation import rank_models
from utils.file_handling import ResultsHandler
from utils.indicators import compute_market_signals, interpret_market_signals
from utils.recommendation import recommend_stock


mts = preprocess_data("pred.csv")
results = {"buy_price": 1000}
ranked_models = rank_models()
current_prices = {ISIN: cp[-1] for ISIN, cp in mts.close_prices.items()}

for position_type in ["short", "long"]:
    top_models = list(
        ranked_models[ranked_models["Position"] == position_type]["Model"].unique()
    )
    results[position_type] = {"Top Models": top_models}
    for optimize in ["risk", "return"]:
        top_stock, predicted_return, model_agreement = recommend_stock(
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

ResultsHandler().write_json_results(results, "recommendation")
