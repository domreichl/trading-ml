import os, json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score

from utils.evaluation import compute_SMAPE, evaluate_sign_predictions
from utils.file_handling import write_csv_results, write_frontend_data


def get_and_process_trades() -> None:
    trades = load_trades_from_database()
    write_csv_results(trades, "trades")
    write_frontend_data(trades, "trades")
    statistics, performance = compute_trading_results(trades)
    write_csv_results(statistics, "trading_statistics")
    write_csv_results(performance, "trading_performance")
    write_frontend_data(statistics, "trading_statistics")


def load_trades_from_database() -> pd.DataFrame:
    db = json.load(open(os.path.join(os.path.dirname(__file__), "config", "db.json")))
    conn_str = f"mysql+pymysql://{db['USER']}:{db['PW']}@{db['HOST']}:{db['PORT']}/{db['SCHEMA']}"
    with create_engine(conn_str).connect() as conn:
        trades = pd.read_sql("select * from trades", con=conn)
    trades.sort_values("ID", inplace=True)
    return trades


def compute_trading_results(df: pd.DataFrame) -> tuple:
    df["NET_PROFIT"] = df["GROSS_PROFIT"] - df["FEES"]
    df["REWARD"] = (df["SELL_PRICE"] - df["BUY_PRICE"]) / df[
        "BUY_PRICE"
    ]  # = ACTUAL_RETURN
    df["ACTUAL_RETURN"] = (df["SELL_PRICE"] * df["SHARES"]) / (
        df["BUY_PRICE"] * df["SHARES"]
    ) - 1
    assert np.array_equal(
        np.round(df["REWARD"].values, 10), np.round(df["ACTUAL_RETURN"].values, 10)
    )
    df["PREDICTED_RETURN"] = (df["PREDICTED_PRICE"] * df["SHARES"]) / (
        df["BUY_PRICE"] * df["SHARES"]
    ) - 1
    df["RETURN_AE"] = np.abs(df["ACTUAL_RETURN"] - df["PREDICTED_RETURN"])
    trading_statistics = compute_trading_statistics(df)
    trading_performance = compute_trading_performance(df)
    return trading_statistics, trading_performance


def compute_trading_statistics(df: pd.DataFrame) -> dict:
    trades_win = df[df["NET_PROFIT"] > 0]
    trades_loss = df[df["NET_PROFIT"] <= 0]
    statistics_dict = {
        "N_TRADES": len(df),
        "N_TRADES_WIN": len(trades_win),
        "N_TRADES_LOSS": len(trades_loss),
        "WIN_RATE": len(trades_win) / len(df) * 100,
        "TOTAL_VOLUME": (df["BUY_PRICE"] * df["SHARES"]).sum(),
        "TOTAL_GROSS_PROFIT": df["GROSS_PROFIT"].sum(),
        "TOTAL_NET_PROFIT": df["NET_PROFIT"].sum(),
        "TOTAL_FEES": df["FEES"].sum(),
        "AVG_VOLUME": (df["BUY_PRICE"] * df["SHARES"]).mean(),
        "AVG_PROFIT": df["NET_PROFIT"].mean(),
        "STD_PROFIT": df["NET_PROFIT"].std(),
        "MAX_WIN": df["NET_PROFIT"].max(),
        "MAX_LOSS": df["NET_PROFIT"].min(),
        "AVG_WIN": trades_win["NET_PROFIT"].mean(),
        "AVG_LOSS": trades_loss["NET_PROFIT"].mean(),
        "SQN": df["REWARD"].mean() / df["REWARD"].std() * np.sqrt(len(df)),
    }
    statistics = pd.DataFrame(columns=statistics_dict.keys())
    statistics.loc[len(statistics)] = statistics_dict
    statistics = statistics.transpose()
    statistics[0] = round(statistics[0], 2)
    statistics = statistics.reset_index()
    statistics.columns = ["Statistic", "Value"]
    return statistics


def compute_trading_performance(df: pd.DataFrame) -> pd.DataFrame:
    performance = {}
    for model in df["MODEL"].unique():
        df_model = df[df["MODEL"] == model]
        buy_price = df_model["BUY_PRICE"].values
        sell_price_gt = df_model["SELL_PRICE"].values
        sell_price_pr = df_model["PREDICTED_PRICE"].values
        win_gt = sell_price_gt > buy_price
        win_pr = sell_price_pr > buy_price
        metrics = evaluate_sign_predictions(win_gt, win_pr)
        performance[model] = {
            "RETURN_MAE": round(np.mean(df_model["RETURN_AE"]), 4),
            "PRICE_SMAPE": round(compute_SMAPE(sell_price_gt, sell_price_pr), 2),
            "ACCURACY": accuracy_score(win_gt, win_pr),
            "PRECISION": metrics["Precision"],
            "RECALL": metrics["Recall"],
            "F1": metrics["F1"],
        }
    performance = pd.DataFrame(performance).transpose()
    performance = performance.reset_index(names="Model")
    return performance


if __name__ == "__main__":
    get_and_process_trades()
