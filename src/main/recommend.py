from utils.data_preprocessing import preprocess_data
from utils.evaluation import rank_models, filter_overfit_models
from utils.file_handling import ResultsHandler
from utils.indicators import compute_market_signals, interpret_market_signals
from utils.recommendation import recommend_stock


results = {}
close_prices = preprocess_data("main.csv").close_prices
ranked_models, relevant_metrics = rank_models()
filtered_models = filter_overfit_models(ranked_models)
relevant_metrics = relevant_metrics[
    relevant_metrics["Model"].isin(filtered_models["Model"])
]
if len(filtered_models) == 0:
    raise Exception("No models left after dropping the overfit ones.")
current_prices = {ISIN: cp[-1] for ISIN, cp in close_prices.items()}

for position_type in ["short", "long"]:
    top_models = list(
        filtered_models[filtered_models["Position"] == position_type]["Model"].unique()
    )
    results[position_type] = {
        "Top Models": top_models,
        "Median SMAPE": relevant_metrics[relevant_metrics["Metric"] == "SMAPE"][
            "Score"
        ].median(),
    }
    if position_type == "short":
        results[position_type]["Median NPV"] = relevant_metrics[
            relevant_metrics["Metric"] == "NPV"
        ]["Score"].median()
    elif position_type == "long":
        results[position_type]["Median PPV"] = relevant_metrics[
            relevant_metrics["Metric"] == "Precision"
        ]["Score"].median()
    for optimize in ["risk", "reward"]:
        top_stock, predicted_return, model_agreement = recommend_stock(
            current_prices, position_type, optimize, top_models=top_models
        )
        trend, state, macdc, rsi, fso, bbb = compute_market_signals(
            close_prices[top_stock]
        )
        trend, state = interpret_market_signals(top_stock, trend, state)
        results[position_type][optimize] = {
            "Top Stock": top_stock,
            "Median Predicted Return": predicted_return,
            "Model Agreement": model_agreement,
            "Market Trend": trend,
            "MACD Crossover": macdc,
            "Market State": state,
            "Relative Strength Index": rsi,
            "Fast Stochastic Oscillator": fso,
            "Bollinger Band Breakout": bbb,
        }

ResultsHandler().write_json_results(results, "recommendation")
