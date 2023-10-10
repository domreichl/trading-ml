import os, pytest
import pandas as pd

from trades import load_trades_from_database, compute_trading_results


@pytest.mark.skip(reason="might fail when IP changes")
def test_trades_load_trades_from_database():
    trades = load_trades_from_database()
    assert len(trades) > 1


def test_trades_compute_statistics():
    statistics, performance = compute_trading_results(
        pd.read_csv(os.path.join(os.path.dirname(__file__), "test_trades.csv"))
    )
    assert len(statistics) == 16
    assert len(performance.columns) == 7
