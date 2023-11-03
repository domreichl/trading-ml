import click, random

from utils.backtests import run_backtests
from utils.config import Config
from utils.data_preparation import prepare_data
from utils.data_preprocessing import preprocess_data
from utils.data_processing import get_df_from_predictions
from utils.evaluation import compute_prediction_performances
from utils.file_handling import ResultsHandler
from utils.indicators import compute_market_signals, interpret_market_signals
from utils.prediction import generate_predictions
from utils.recommendation import recommend_stock, recommend_close_position
from utils.trades import get_and_process_trades
from utils.training import train_model
from utils.validation import validate_model
from utils.visualization import (
    plot_price_predictions,
    plot_test_metrics,
    plot_validation_metrics,
    plot_price_forecast,
)


@click.group()
def cli():
    pass


@cli.command()
def prepare():
    cfg = Config()
    cfg.set_dates("2000-01-03", "2023-10-31")
    prepare_data("exp.csv", cfg)


@cli.command()
@click.argument("model_name")
def train(model_name: str):
    train_model("exp_" + model_name, mts=preprocess_data("exp.csv"))


@cli.command()
@click.argument("model_name")
def validate(model_name: str):
    mts = preprocess_data("exp.csv", model_name=model_name)
    model_name = "exp_" + model_name
    rmse, ps = validate_model(model_name, mts, n_validations=10)
    print(f"Validation results for {model_name}:")
    print("RMSE: ", rmse)
    print("PredictiveScore: ", ps)


@cli.command()
@click.argument("model_name")
def test(model_name: str):
    mts = preprocess_data("exp.csv", model_name=model_name)
    model_name = "exp_" + model_name
    returns_predicted, prices_predicted = generate_predictions(model_name, mts)
    df = compute_prediction_performances(
        mts.get_test_returns(),
        returns_predicted,
        mts.get_test_prices(),
        prices_predicted,
        mts.get_naive_error(),
        model_name,
    )
    print(df)


@cli.command()
@click.argument("model_name")
@click.argument("ts_name", required=False)
def predict(model_name: str, ts_name: str = ""):
    mts = preprocess_data("exp.csv", model_name=model_name)
    model_name = "exp_" + model_name
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
@click.argument("ts_name", required=False)
def forecast(ts_name: str = ""):
    forecast = ResultsHandler().load_csv_results("forecast")
    if not ts_name:
        ts_name = random.choice(list(forecast["ISIN"].unique()))
    forecast = forecast[forecast["ISIN"] == ts_name].reset_index(drop=True)
    print(forecast)
    plot_price_forecast(forecast)


@cli.command()
@click.argument("position_type")
@click.argument("optimize")
def recommend_open(position_type: str, optimize: str):
    close_prices = preprocess_data("main.csv").close_prices
    current_prices = {ts_name: cp[-1] for ts_name, cp in close_prices.items()}
    top_stock, _, _ = recommend_stock(current_prices, position_type, optimize)
    trend, state, macdc, rsi, fso, bbb = compute_market_signals(close_prices[top_stock])
    interpret_market_signals(top_stock, trend, state)
    print(" - MACD Crossover:", macdc)
    print(" - Relative Strength Index:", rsi)
    print(" - Fast Stochastic Oscillator:", fso)
    print(" - Bollinger Band Breakout:", bbb)


@cli.command()
@click.argument("position_type")
@click.argument("ts_name")
def recommend_close(position_type: str, ts_name: str):
    close_prices = preprocess_data("main.csv").close_prices
    current_price = close_prices[ts_name][-1]
    forecast = ResultsHandler().load_csv_results("forecast")
    forecast = forecast[forecast["ISIN"] == ts_name]
    recommend_close_position(forecast, current_price, position_type)
    trend, state, macdc, rsi, fso, bbb = compute_market_signals(close_prices[ts_name])
    interpret_market_signals(ts_name, trend, state)
    print(" - MACD Crossover:", macdc)
    print(" - Relative Strength Index:", rsi)
    print(" - Fast Stochastic Oscillator:", fso)
    print(" - Bollinger Band Breakout:", bbb)


@cli.command()
@click.argument("metrics_type")
def plot_metrics(metrics_type: str):
    if metrics_type == "validation":
        plot_validation_metrics()
    elif metrics_type == "test":
        plot_test_metrics()


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
