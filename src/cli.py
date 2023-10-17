import click, random

from data_preparation import prepare_data
from backtesting import run_backtests
from prediction import generate_predictions
from recommendation_close import recommend_close_position
from recommendation_open import recommend_stock
from validation import validate_model
from visualization import (
    plot_prediction_metrics,
    plot_optimization_metrics,
    plot_price_predictions,
    plot_validation_metrics,
    plot_price_forecast,
)

from utils.data_preprocessing import preprocess_data
from utils.data_processing import get_df_from_predictions, get_forecast_df
from utils.evaluation import compute_prediction_performances
from utils.indicators import compute_market_signals, print_market_signals
from utils.model_selection import pick_top_models


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
    mts = preprocess_data()
    mae, rmse = validate_model(model_name, mts)
    print(f"Validation results for {model_name}:")
    print("MAE: ", mae)
    print("RMSE: ", rmse)


@cli.command()
@click.argument("model_name")
def evaluate(model_name: str):
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
    mts = preprocess_data()
    mts.merge_features(for_deep_learning=deep_learning)
    model_name = "prod_" + model_name
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
    df = df[df["ISIN"] == ts_name]
    print(df)
    plot_price_forecast(df)


@cli.command()
@click.argument("position_type")
@click.argument("optimize")
def recommend_open(position_type: str, optimize: str):
    mts = preprocess_data()
    current_prices = {ts_name: cp[-1] for ts_name, cp in mts.close_prices.items()}
    top_models = pick_top_models(position_type)
    top_stock = recommend_stock(top_models, current_prices, position_type, optimize)
    overbought, bullish = compute_market_signals(mts.close_prices[top_stock])
    print_market_signals(top_stock, overbought, bullish)


@cli.command()
@click.argument("position_type")
@click.argument("ts_name")
def recommend_close(position_type: str, ts_name: str):
    mts = preprocess_data()
    current_price = mts.close_prices[ts_name][-1]
    recommend_close_position(ts_name, current_price, position_type)
    overbought, bullish = compute_market_signals(mts.close_prices[ts_name])
    print_market_signals(ts_name, overbought, bullish)


for cmd in [
    prepare,
    backtest,
    validate,
    evaluate,
    plot_metrics,
    predict,
    forecast,
    recommend_open,
    recommend_close,
]:
    cli.add_command(cmd)
