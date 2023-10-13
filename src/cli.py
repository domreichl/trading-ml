import click
import pandas as pd

from config.config import data_config, paths
from utils.data_preprocessing import preprocess_data

from data_preparation import prepare_data
from backtesting import run_backtests
from prediction import (
    generate_predictions,
    compute_prediction_performances,
    get_df_from_predictions,
)
from validation import validate_model
from visualization import (
    plot_prediction_metrics,
    plot_optimization_metrics,
    plot_price_predictions,
    plot_validation_metrics,
)


@click.group()
def cli():
    pass


@cli.command()
def prepare():
    prepare_data()


@cli.command()
def backtest():
    run_backtests()


@cli.command()
@click.argument("model_name")
def validate(model_name: str):
    mts = preprocess_data(
        paths["csv"], data_config["look_back_window_size"], include_stock_index=True
    )
    mae, rmse = validate_model(model_name, mts)
    print(f"Validation results for {model_name}:")
    print("MAE: ", mae)
    print("RMSE: ", rmse)


@cli.command()
@click.argument("model_name")
@click.argument("ts_name", required=False)
def predict(model_name: str, ts_name: str):
    mts = preprocess_data(
        paths["csv"],
        data_config["look_back_window_size"],
        include_stock_index=True,
    )
    returns_predicted, prices_predicted = generate_predictions(model_name, mts)
    df = get_df_from_predictions(
        mts.get_test_returns(),
        returns_predicted,
        mts.get_test_prices(),
        prices_predicted,
        mts.get_test_dates(),
        model_name,
    )
    print(df)
    plot_price_predictions(df, ts_name)


@cli.command()
@click.argument("model_name")
def evaluate(model_name: str):
    mts = preprocess_data(
        paths["csv"], data_config["look_back_window_size"], include_stock_index=True
    )
    returns_predicted, prices_predicted = generate_predictions(model_name, mts)
    df = compute_prediction_performances(
        mts.get_test_returns(),
        returns_predicted,
        mts.get_test_prices(),
        prices_predicted,
        mts.get_naive_errors(),
        model_name,
    )
    print(df)


@cli.command()
@click.argument("metrics_type")
def plot_metrics(metrics_type: str):
    if metrics_type == "optimization":
        plot_optimization_metrics()
    elif metrics_type == "validation":
        plot_validation_metrics()
    elif metrics_type == "evaluation":
        plot_prediction_metrics()


for cmd in [prepare, backtest, validate, predict, evaluate, plot_metrics]:
    cli.add_command(cmd)
