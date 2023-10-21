import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from utils.data_preprocessing import preprocess_data
from utils.data_processing import (
    stack_array_from_dict,
    get_signs_from_prices,
    get_final_predictions_from_dict,
)
from utils.evaluation import (
    evaluate_sign_predictions,
    evaluate_price_predictions,
    compute_SMAPE,
    evaluate_return_predictions,
    process_metrics,
)
from utils.file_handling import DataHandler
from utils.trades import load_trades_from_database, compute_trading_results


def test_data_classes_mts_merge_features():
    mts = preprocess_data(Path(__file__).parent.joinpath("test_data.csv"))
    assert mts.x_train.shape == (4929, 260, 4)
    assert mts.y_train.shape == (4929, 10, 4)
    mts.merge_features()
    assert mts.x_train.shape == (4930, 260, 4)
    assert mts.y_train.shape == (0,)
    assert mts.x_test.shape == (0,)
    assert mts.y_test.shape == (10, 4)


def test_data_classes_mts_merge_features_dl():
    mts = preprocess_data(Path(__file__).parent.joinpath("test_data.csv"))
    mts.merge_features(for_deep_learning=True)
    assert mts.x_train.shape == (4930, 260, 4)
    assert mts.y_train.shape == (4930, 10, 4)


def test_data_classes_mts_get_forecast_dates():
    mts = preprocess_data(Path(__file__).parent.joinpath("test_data.csv"))
    forecast_dates = mts.get_forecast_dates()
    assert mts.dates[-1] == "2023-09-15"
    assert forecast_dates[0] == "2023-09-18"
    assert forecast_dates[-1] == "2023-09-29"


def test_data_preprocessing():
    mts = preprocess_data(Path(__file__).parent.joinpath("test_data.csv"))
    assert round(mts.x_train[-1][-1][-1], 4) == 0.4173
    assert round(mts.y_train[0][0][0], 4) == 0.8725
    assert round(mts.x_test[-1][-1], 4) == 0.4042
    assert round(mts.y_test[1][1], 4) == 0.4069
    assert mts.x_train.shape == (4929, 52 * 5, 4)
    assert mts.y_train.shape == (4929, 10, 4)
    assert mts.x_test.shape == (52 * 5, 4)
    assert mts.y_test.shape == (10, 4)
    assert len(mts.names) == len(mts.close_prices.keys()) == 4
    assert len(mts.log_returns) == 4


def test_data_processing_stack_array_from_dict():
    dictionary = {"A": [0, 1, 2], "B": [3, 4, 5]}
    array = stack_array_from_dict(dictionary, 0)
    assert array.shape == (2, 3)
    array = stack_array_from_dict(dictionary, 1)
    assert array.shape == (3, 2)


def test_data_processing_get_signs_from_prices():
    prices = {"A": [10, 11, 12], "B": [12, 11, 10]}
    array = get_signs_from_prices(prices)
    assert np.array_equal(array, [1, 1, 0, 0]) == True


def test_data_processing_get_final_predictions_from_dict():
    dictionary = {"A": [10, 11, 12], "B": [13, 14, 15]}
    array = get_final_predictions_from_dict(dictionary)
    assert np.array_equal(array, [12, 15]) == True


def test_evaluation_evaluate_sign_predictions():
    gt = np.array([[0, 1, 0], [1, 1, 0]])
    pr = np.array([[0, 1, 1], [1, 1, 1]])
    P = len(gt[np.where(gt == 1)])
    Ppr = len(pr[np.where(pr == 1)])
    Npr = len(pr[np.where(pr == 0)])
    TP = len(pr[np.where((pr == 1) & (pr == gt))])
    TN = len(pr[np.where((pr == 0) & (pr == gt))])
    precision = TP / Ppr
    recall = TP / P
    f1_score = 2 * (precision * recall) / (precision + recall)
    negative_predictive_value = TN / Npr
    metrics = evaluate_sign_predictions(gt.reshape(-1, 1), pr.reshape(-1, 1))
    assert metrics["Precision"] == precision
    assert metrics["Recall"] == recall
    assert metrics["F1"] == f1_score
    assert metrics["NPV"] == negative_predictive_value


def test_evaluation_evaluate_price_predictions():
    gt = np.array([100, 100])
    pr = np.array([90, 110])
    metrics = evaluate_price_predictions(gt, pr)
    assert round(metrics["SMAPE"], 2) == 5.01


def test_evaluation_compute_SMAPE():
    gt = np.array([100, 100])
    pr = np.array([90, 90])
    smape = compute_SMAPE(gt, pr)
    assert round(smape, 2) == 5.26


def test_evaluation_evaluate_return_predictions():
    gt = np.array([100, 100])
    pr = np.array([90, 110])
    naive_errors = (9, 15)
    metrics = evaluate_return_predictions(gt, pr, naive_errors)
    assert metrics["MAE"] == 10
    assert metrics["RMSE"] == 10
    assert round(metrics["MASE"], 2) == 1.11
    assert round(metrics["RMSSE"], 2) == 0.67


def test_evaluation_process_metrics():
    metrics = {"MetricA": 1, "MetricB": 2, "MetricC": 3}
    performance = process_metrics(metrics, "someTarget", "someModel")
    assert len(performance) == 3
    assert int(performance["Score"].loc[performance["Metric"] == "MetricB"][1]) == 2


#@pytest.mark.skip(reason="might fail when IP changes")
def test_trades_load_trades_from_database():
    trades = load_trades_from_database()
    assert len(trades) > 1


def test_trades_compute_statistics():
    statistics, performance = compute_trading_results(
        DataHandler().load_csv_data(Path(__file__).parent.joinpath("test_trades.csv"))
    )
    assert len(statistics) == 16
    assert len(performance.columns) == 7
