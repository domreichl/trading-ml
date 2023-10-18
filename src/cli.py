import click, random

from pipeline.prepare import prepare_data
from pipeline.train import train_model
from pipeline.validate import validate_model
from pipeline.select import pick_top_models
from utils.backtests import run_backtests
from utils.data_preprocessing import preprocess_data
from utils.data_processing import get_df_from_predictions, get_forecast_df
from utils.evaluation import compute_prediction_performances
from utils.indicators import compute_market_signals, interpret_market_signals
from utils.prediction import generate_predictions
from utils.recommendation import recommend_stock, recommend_close_position
from utils.trades import get_and_process_trades
from utils.visualization import (
    plot_optimization_metrics,
    plot_price_predictions,
    plot_test_performance,
    plot_validation_metrics,
    plot_price_forecast,
)


@click.group()
def cli():
    pass


@cli.command()
def prepare():
    prepare_data()


@cli.command()
@click.argument("model_name")
def train(model_name: str):
    model_name = "cli_" + model_name
    train_model(model_name, mts=preprocess_data())


@cli.command()
@click.argument("model_name")
def validate(model_name: str):
    model_name = "cli_" + model_name
    mts = preprocess_data()
    mae, rmse = validate_model(model_name, mts)
    print(f"Validation results for {model_name}:")
    print("MAE: ", mae)
    print("RMSE: ", rmse)


@cli.command()
@click.argument("model_name")
def test(model_name: str):
    model_name = "cli_" + model_name
    mts = preprocess_data()
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
@click.argument("model_name")
@click.argument("ts_name", required=False)
def predict(model_name: str, ts_name: str = ""):
    model_name = "cli_" + model_name
    mts = preprocess_data()
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
    df = df[df["ISIN"] == ts_name].reset_index(drop=True)
    print(df)
    plot_price_predictions(df)


@cli.command()
@click.argument("model_name")
@click.argument("ts_name", required=False)
def forecast(model_name: str, ts_name: str = ""):
    model_name = "prod_" + model_name
    deep_learning = False
    if "lstm" in model_name:
        deep_learning = True
    mts = preprocess_data()
    mts.merge_features(for_deep_learning=deep_learning)
    returns_predicted, prices_predicted = generate_predictions(
        model_name, mts, forecast=True
    )
    df = get_forecast_df(
        returns_predicted,
        prices_predicted,
        mts.get_forecast_dates(),
        model_name,
    )
    if not ts_name:
        ts_name = random.choice(list(df["ISIN"].unique()))
    df = df[df["ISIN"] == ts_name].reset_index(drop=True)
    print(df)
    plot_price_forecast(df)


@cli.command()
@click.argument("position_type")
@click.argument("optimize")
def recommend_open(position_type: str, optimize: str):
    mts = preprocess_data()
    current_prices = {ts_name: cp[-1] for ts_name, cp in mts.close_prices.items()}
    top_models = pick_top_models(position_type, prod=True)
    top_stock, _, _ = recommend_stock(
        top_models, current_prices, position_type, optimize
    )
    trend, state = compute_market_signals(mts.close_prices[top_stock])
    interpret_market_signals(top_stock, trend, state)


@cli.command()
@click.argument("position_type")
@click.argument("ts_name")
def recommend_close(position_type: str, ts_name: str):
    mts = preprocess_data()
    current_price = mts.close_prices[ts_name][-1]
    recommend_close_position(ts_name, current_price, position_type)
    trend, state = compute_market_signals(mts.close_prices[ts_name])
    interpret_market_signals(ts_name, trend, state)


@cli.command()
@click.argument("metrics_type")
def plot_metrics(metrics_type: str):
    if metrics_type == "optimization":
        plot_optimization_metrics()
    elif metrics_type == "validation":
        plot_validation_metrics()
    elif metrics_type == "evaluation":
        plot_test_performance()


@cli.command()
def backtest():
    run_backtests()


@cli.command()
def fetch_trades():
    get_and_process_trades()


for cmd in [
    prepare,
    train,
    validate,
    test,
    predict,
    forecast,
    recommend_open,
    recommend_close,
    plot_metrics,
    backtest,
    fetch_trades,
]:
    cli.add_command(cmd)
