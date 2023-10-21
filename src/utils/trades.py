import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

from utils.file_handling import ResultsHandler


def get_and_process_trades() -> None:
    rh = ResultsHandler()
    trades = load_trades_from_database()
    statistics = compute_trading_statistics(trades)
    rh.write_csv_results(trades, "trades")
    rh.write_json_results(statistics, "trades_statistics")
    rh.write_frontend_data(trades, "trades")
    rh.write_frontend_data(statistics, "trades_statistics")


def load_trades_from_database() -> pd.DataFrame:
    db = yaml.safe_load(
        open(Path(__file__).parent.parent.parent.joinpath("config", "db.yaml"))
    )
    conn_str = f"mysql+pymysql://{db['USER']}:{db['PW']}@{db['HOST']}:{db['PORT']}/{db['SCHEMA']}"
    with create_engine(conn_str).connect() as conn:
        trades = pd.read_sql("select * from trades", con=conn)
    trades.sort_values("ID", inplace=True)
    return trades


def compute_trading_statistics(df: pd.DataFrame) -> dict:
    df["NET_PROFIT"] = df["GROSS_PROFIT"] - df["FEES"]
    df["REWARD"] = (df["CLOSE_PRICE"] - df["OPEN_PRICE"]) / df["OPEN_PRICE"]
    trades_win = df[df["NET_PROFIT"] > 0]
    trades_loss = df[df["NET_PROFIT"] <= 0]
    statistics_dict = {
        "N_TRADES": len(df),
        "N_TRADES_WIN": len(trades_win),
        "N_TRADES_LOSS": len(trades_loss),
        "WIN_RATE": round(len(trades_win) / len(df) * 100),
        "TOTAL_VOLUME": (df["OPEN_PRICE"] * df["SHARES"]).sum(),
        "TOTAL_GROSS_PROFIT": df["GROSS_PROFIT"].sum(),
        "TOTAL_NET_PROFIT": df["NET_PROFIT"].sum(),
        "TOTAL_FEES": df["FEES"].sum(),
        "AVG_VOLUME": round((df["OPEN_PRICE"] * df["SHARES"]).mean(), 2),
        "AVG_PROFIT": round(df["NET_PROFIT"].mean(), 2),
        "STD_PROFIT": round(df["NET_PROFIT"].std(), 2),
        "MAX_WIN": df["NET_PROFIT"].max(),
        "MAX_LOSS": df["NET_PROFIT"].min(),
        "AVG_WIN": round(trades_win["NET_PROFIT"].mean(), 2),
        "AVG_LOSS": round(trades_loss["NET_PROFIT"].mean(), 2),
        "SQN": df["REWARD"].mean() / df["REWARD"].std() * np.sqrt(len(df)),
    }
    return statistics_dict
