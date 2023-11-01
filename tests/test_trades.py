import pytest
from pathlib import Path

from utils.file_handling import DataHandler
from utils.trades import load_trades_from_database, compute_trading_statistics


@pytest.mark.skip(reason="might fail when IP changes")
def test_load_trades_from_database():
    trades = load_trades_from_database()
    assert len(trades) > 1


def test_compute_statistics():
    statistics = compute_trading_statistics(
        DataHandler().load_csv_data(Path(__file__).parent.joinpath("test_trades.csv"))
    )
    assert len(statistics) == 16
