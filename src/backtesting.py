import numpy as np

from config.config import data_config, paths
from utils.data_preprocessing import preprocess_data
from utils.data_processing import stack_array_from_dict
from utils.backtest_calc import (
    compute_expected_buy_and_hold_profits,
    compute_expected_weekly_trading_profits,
)
from utils.backtest_plot import (
    visualize_averaged_yearly_log_returns,
    visualize_expected_buy_and_hold_profits,
    visualize_expected_weekly_trading_profits,
)


def run_backtests(buy_price: float, buy_fee: float, start_year: int) -> None:
    mts = preprocess_data(
        paths["csv"],
        look_back_window_size=data_config["look_back_window_size"],
        include_stock_index=False,
    )
    log_returns = stack_array_from_dict(mts.log_returns, 0)
    returns = np.exp(log_returns)
    visualize_averaged_yearly_log_returns(log_returns, start_year)
    profits_bh = compute_expected_buy_and_hold_profits(returns, buy_price, buy_fee)
    visualize_expected_buy_and_hold_profits(*profits_bh, buy_price)
    profits_wt = compute_expected_weekly_trading_profits(returns, buy_price, buy_fee)
    visualize_expected_weekly_trading_profits(profits_wt, buy_price)


if __name__ == "__main__":
    run_backtests(1000, 0.99, data_config["start_date"].year)
