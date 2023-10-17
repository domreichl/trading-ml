import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from config.model_config import model_config
from utils.data_processing import (
    get_signs_from_prices,
    stack_array_from_dict,
    get_final_predictions_from_dict,
)


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
            compute_sign_prediction_performance(
                prices_actual, prices_predicted, model_name
            ),
            compute_return_prediction_performance(
                returns_actual, returns_predicted, naive_errors, model_name
            ),
            compute_price_prediction_performance(
                prices_actual, prices_predicted, model_name
            ),
        ]
    )


def compute_sign_prediction_performance(
    prices_actual: dict, prices_predicted: dict, model_name: str
) -> pd.DataFrame:
    performance = process_metrics(
        evaluate_sign_predictions(
            get_signs_from_prices(prices_actual),
            get_signs_from_prices(prices_predicted),
        ),
        "Sign",
        model_name,
    )
    return performance


def compute_return_prediction_performance(
    returns_actual: dict,
    returns_predicted: dict,
    naive_errors: tuple[float, float],
    model_name: str,
) -> pd.DataFrame:
    performance = process_metrics(
        evaluate_return_predictions(
            stack_array_from_dict(returns_actual, 1),
            stack_array_from_dict(returns_predicted, 1),
            naive_errors,
        ),
        "Return",
        model_name,
    )
    return performance


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


def evaluate_sign_predictions(gt: np.array, pr: np.array) -> dict:
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        gt, pr, average="binary"
    )
    negative_predictive_value, _, _, _ = precision_recall_fscore_support(
        1 - gt, 1 - pr, average="binary"
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


def evaluate_return_predictions(
    gt: np.array, pr: np.array, naive_errors: tuple[float, float] = None
) -> dict:
    mae = np.mean(np.abs(pr - gt))  # for homogenous returns scales
    rmse = np.sqrt(np.mean(np.square(pr - gt)))  # to penalize large errors
    metrics = {"MAE": mae, "RMSE": rmse}
    if naive_errors:
        metrics["MASE"] = mae / naive_errors[0]
        metrics["RMSSE"] = rmse / naive_errors[1]
    return metrics


def process_metrics(
    metrics: dict,
    target_name: str,
    model_name: str,
) -> pd.DataFrame:
    performance = pd.DataFrame(columns=model_config["results_cols"])
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
