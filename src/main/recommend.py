from utils.data_preprocessing import preprocess_data
from utils.file_handling import ResultsHandler
from utils.indicators import compute_market_signals, interpret_market_signals
from utils.recommendation import recommend_stock


rh = ResultsHandler()
results = {}
close_prices = preprocess_data("main.csv").close_prices
model_ratings = rh.load_json_results("test_ratings")
top_models = list(model_ratings.keys())
metrics = rh.load_csv_results("test_metrics")
metrics = metrics[metrics["Model"].isin(top_models)]
current_prices = {ISIN: cp[-1] for ISIN, cp in close_prices.items()}

results = {
    "TopModelRatings": model_ratings,
    "MedianTopModelTestMetrics": {
        k: metrics[metrics["Metric"] == k]["Score"].median()
        for k in metrics["Metric"].unique()
    },
}

for position_type in ["short", "long"]:
    results[position_type] = {}
    for optimize in ["risk", "reward"]:
        top_stock, predicted_return, model_agreement = recommend_stock(
            current_prices, position_type, optimize, top_models=top_models
        )
        trend, state, macdc, rsi, fso, bbb = compute_market_signals(
            close_prices[top_stock]
        )
        trend, state = interpret_market_signals(top_stock, trend, state)
        results[position_type][optimize] = {
            "TopStock": top_stock,
            "MedianPredictedReturn": predicted_return,
            "ModelAgreement": model_agreement,
            "MarketTrend": trend,
            "MACDCrossover": macdc,
            "MarketState": state,
            "RelativeStrengthIndex": rsi,
            "FastStochasticOscillator": fso,
            "BollingerBandBreakout": bbb,
        }

ResultsHandler().write_json_results(results, "recommendation")
