import click, random

from config.config import data_config, paths
from utils.data_preprocessing import preprocess_data

from data_preparation import prepare_data
from backtesting import run_backtests
from prediction import generate_predictions
from validation import validate_model
from visualization import (
    plot_prediction_metrics,
    plot_optimization_metrics,
    plot_price_predictions,
    plot_validation_metrics,
    plot_price_forecast,
)
from utils.data_processing import get_df_from_predictions, get_forecast_df
from utils.evaluation import compute_prediction_performances


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


@cli.command()
@click.argument("model_name")
@click.argument("ts_name", required=False)
def predict(model_name: str, ts_name: str = ""):
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
    if not ts_name:
        ts_name = random.choice(list(df["ISIN"].unique()))
    df = df[df["ISIN"] == ts_name]
    print(df)
    plot_price_predictions(df)


@cli.command()
@click.argument("model_name")
@click.argument("ts_name", required=False)
def forecast(model_name: str, ts_name: str = ""):
    deep_learning = False
    if model_name == "lstm":
        deep_learning = True
    mts = preprocess_data(
        paths["csv"],
        look_back_window_size=data_config["look_back_window_size"],
        include_stock_index=True,
    )
    mts.merge_features(for_deep_learning=deep_learning)
    model_name = "prod_" + model_name
    returns_predicted, prices_predicted = generate_predictions(model_name, mts)
    df = get_forecast_df(
        returns_predicted,
        prices_predicted,
        mts.get_forecast_dates(),
        model_name,
    )
    if not ts_name:
        ts_name = random.choice(list(df["ISIN"].unique()))
    df = df[df["ISIN"] == ts_name]
    print(df)
    plot_price_forecast(df)


for cmd in [prepare, backtest, validate, evaluate, plot_metrics, predict, forecast]:
    cli.add_command(cmd)
