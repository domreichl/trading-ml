import numpy as np

from utils.evaluation import (
    evaluate_sign_predictions,
    evaluate_price_predictions,
    compute_SMAPE,
    evaluate_return_predictions,
    process_metrics,
)


def test_evaluate_sign_predictions():
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


def test_evaluate_price_predictions():
    gt = np.array([100, 100])
    pr = np.array([90, 110])
    metrics = evaluate_price_predictions(gt, pr)
    assert round(metrics["SMAPE"], 2) == 5.01


def test_compute_SMAPE():
    gt = np.array([100, 100])
    pr = np.array([90, 90])
    smape = compute_SMAPE(gt, pr)
    assert round(smape, 2) == 5.26


def test_evaluate_return_predictions():
    gt = np.array([100, 100])
    pr = np.array([90, 110])
    naive_errors = (9, 15)
    metrics = evaluate_return_predictions(gt, pr, naive_errors)
    assert metrics["MAE"] == 10
    assert metrics["RMSE"] == 10
    assert round(metrics["MASE"], 2) == 1.11
    assert round(metrics["RMSSE"], 2) == 0.67


def test_process_metrics():
    metrics = {"MetricA": 1, "MetricB": 2, "MetricC": 3}
    performance = process_metrics(metrics, "someTarget", "someModel")
    assert len(performance) == 3
    assert int(performance["Score"].loc[performance["Metric"] == "MetricB"][1]) == 2
