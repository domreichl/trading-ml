import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from typing import Optional

from utils.data_processing import (
    get_final_predictions_from_dict,
    get_signs_from_returns,
    stack_array_from_dict,
)
from utils.file_handling import ResultsHandler


def rank_models(top_n: int = 3) -> pd.DataFrame:
    test_metrics = ResultsHandler().load_csv_results("test_metrics")
    test_metrics = test_metrics[
        test_metrics["Metric"].isin(["Precision", "NPV", "SMAPE"])
    ]
    position_types, top_models, ranks = [], [], []
    for position_type in ["long", "short"]:
        sorted_ratings = rate_models(test_metrics, position_type)
        for i, model_name in enumerate(list(sorted_ratings.keys())):
            if i == top_n:
                break
            position_types.append(position_type)
            top_models.append(model_name)
            ranks.append(i + 1)
    if max(ranks) < top_n:
        (f"Warning: Only {len(top_models)} top models were ranked.")
    return pd.DataFrame(
        {"Position": position_types, "Rank": ranks, "Model": top_models}
    )


def rate_models(test_metrics: pd.DataFrame, position_type: str) -> dict:
    ratings = {}
    for model_name in test_metrics["Model"].unique():
        metrics = test_metrics[test_metrics["Model"] == model_name]
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
        f"\nRatings of top {test_metrics['Model'].nunique()} validated models ({position_type} position):"
    )
    [print(f" [{i+1}] {k}: {v}") for i, (k, v) in enumerate(sorted_ratings.items())]
    return sorted_ratings


def compute_prediction_performances(
    returns_actual: dict,
    returns_predicted: dict,
    prices_actual: dict,
    prices_predicted: dict,
    naive_errors: tuple[float, float],
    model_name: str,
):
    return pd.concat(
        [
            compute_return_prediction_performance(
                returns_actual, returns_predicted, naive_errors, model_name
            ),
            compute_price_prediction_performance(
                prices_actual, prices_predicted, model_name
            ),
        ]
    )


def compute_return_prediction_performance(
    returns_actual: dict,
    returns_predicted: dict,
    naive_errors: tuple[float, float],
    model_name: str,
) -> pd.DataFrame:
    gt = stack_array_from_dict(returns_actual, 1)
    pr = stack_array_from_dict(returns_predicted, 1)
    sign_performance = process_metrics(
        evaluate_sign_predictions(
            get_signs_from_returns(gt), get_signs_from_returns(pr)
        ),
        "Sign",
        model_name,
    )
    performance = process_metrics(
        evaluate_return_predictions(gt, pr, naive_errors),
        "Return",
        model_name,
    )
    return pd.concat([sign_performance, performance])


def compute_price_prediction_performance(
    prices_actual: dict, prices_predicted: dict, model_name: str
) -> pd.DataFrame:
    performance = process_metrics(
        evaluate_price_predictions(
            get_final_predictions_from_dict(prices_actual),
            get_final_predictions_from_dict(prices_predicted),
        ),
        "Price",
        model_name,
    )
    return performance


def evaluate_return_predictions(
    gt: np.array, pr: np.array, naive_errors: Optional[tuple[float, float]] = None
) -> dict:
    metrics = {
        "MAE": np.mean(np.abs(pr - gt)),  # for homogenous returns scales
        "RMSE": np.sqrt(np.mean(np.square(pr - gt))),  # to penalize large errors
    }
    if naive_errors:
        metrics["MASE"] = metrics["MAE"] / naive_errors[0]
        metrics["RMSSE"] = metrics["RMSE"] / naive_errors[1]
    return metrics


def evaluate_sign_predictions(gt: np.array, pr: np.array) -> dict:
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        gt, pr, average="binary", zero_division=1.0
    )
    negative_predictive_value, _, _, _ = precision_recall_fscore_support(
        1 - gt, 1 - pr, average="binary", zero_division=1.0
    )
    metrics = dict(
        zip(
            ["Precision", "Recall", "F1", "NPV"],
            [precision, recall, f1_score, negative_predictive_value],
        )
    )
    return metrics


def evaluate_price_predictions(gt: np.array, pr: np.array) -> dict:
    # SMAPE because absolute prices vary between series
    return {"SMAPE": compute_SMAPE(gt, pr)}


def compute_SMAPE(gt: np.array, pr: np.array) -> float:
    # Symmetric mean absolute percentage error without division by 2 in the denominator to keep results between 0% and 100%
    return float(
        100 / len(gt) * np.sum(np.abs(pr - gt) / np.add(np.abs(gt), np.abs(pr)))
    )


def process_metrics(
    metrics: dict,
    target_name: str,
    model_name: str,
) -> pd.DataFrame:
    performance = pd.DataFrame(columns=["Model", "Target", "Metric", "Score"])
    for name, score in metrics.items():
        performance.loc[len(performance)] = [
            model_name,
            target_name,
            name,
            round(score, 4),
        ]
    return performance


# def monte_carlo_random_walk(mts: object, n_days: int, n_simulations: int) -> dict:
#     """
#     This function serves no purpose yet.
#     As n_simulations approaches infinity, the mean of simulated returns tends to 1.
#     Compounding is negligible if n_days is small.
#     """

#     volatilities = np.std(
#         mts.get_returns_from_features(np.concatenate([mts.y_train, mts.y_test], 0)), 0
#     )
#     assert len(volatilities) == len(mts.names)

#     simulations = {}
#     for i, ISIN in enumerate(mts.names):
#         simulations_i = []
#         for _ in range(n_simulations):
#             simulations_i.append(
#                 [1 + np.random.normal(0, volatilities[i]) for day in range(n_days)]
#             )
#         simulations[ISIN] = np.mean(np.stack(simulations_i, 0), 0)

#     return simulations
