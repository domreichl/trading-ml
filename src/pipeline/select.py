import pandas as pd

from utils.file_handling import ResultsHandler


def pick_top_models(position_type: str, prod: bool = False) -> str:
    top_models = pick_top_models_validation()
    top_models = pick_top_models_rating(top_models, position_type)
    if prod:
        top_models = [name.replace("eval_", "prod_") for name in top_models]
    return top_models


def pick_top_models_validation(n: int = 5) -> list[str]:
    validation = ResultsHandler().load_csv_results("validation")
    validation.sort_values("RMSE", inplace=True)
    return list(validation["Model"][:n])


def pick_top_models_rating(
    top_models: list[str], position_type: str, n: int = 3
) -> list[str]:
    ratings = {}
    performance = ResultsHandler().load_csv_results("test_performance")
    performance.drop(columns=["Target"], inplace=True)
    performance = performance[
        (performance["Model"].isin(top_models))
        & (performance["Metric"].isin(["Precision", "NPV", "SMAPE"]))
    ]
    for model_name in top_models:
        metrics = performance[performance["Model"] == model_name]
        if position_type == "long":
            sign_prediction_score = metrics[metrics["Metric"] == "Precision"][
                "Score"
            ].iloc[0]
        elif position_type == "short":
            sign_prediction_score = metrics[metrics["Metric"] == "NPV"]["Score"].iloc[0]
        ratings[model_name] = round(
            sign_prediction_score
            / metrics[metrics["Metric"] == "SMAPE"]["Score"].iloc[0]
            * 100,
            2,
        )
    sorted_ratings = dict(
        sorted(ratings.items(), key=lambda item: item[1], reverse=True)
    )
    print(
        f"\nRatings of top {len(top_models)} validated models ({position_type} position):"
    )
    [print(f" [{i+1}] {k}: {v}") for i, (k, v) in enumerate(sorted_ratings.items())]
    top_models = list(sorted_ratings.keys())[:n]
    return top_models


if __name__ == "__main__":
    position_types, top_models, ranks = [], [], []
    for position_type in ["long", "short"]:
        for i, top_model in enumerate(pick_top_models(position_type)):
            position_types.append(position_type)
            top_models.append(top_model)
            ranks.append(i + 1)
    df = pd.DataFrame({"Position": position_types, "Rank": ranks, "Model": top_models})
    ResultsHandler().write_csv_results(df, "selection")
