import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.data_preprocessing import preprocess_data
from utils.data_processing import stack_array_from_dict
from utils.file_handling import ResultsHandler


def run_backtests(
    buy_price: float = 1000, buy_fee: float = 0.99, start_year: int = 2000
) -> None:
    mts = preprocess_data()
    log_returns = stack_array_from_dict(mts.log_returns, 0)
    returns = np.exp(log_returns)
    visualize_averaged_yearly_log_returns(log_returns, start_year)
    profits_bh = compute_expected_buy_and_hold_profits(returns, buy_price, buy_fee)
    visualize_expected_buy_and_hold_profits(*profits_bh, buy_price)
    profits_wt = compute_expected_weekly_trading_profits(returns, buy_price, buy_fee)
    ResultsHandler().write_frontend_data(profits_wt, "backtest")
    visualize_expected_weekly_trading_profits(profits_wt, buy_price)


def compute_expected_buy_and_hold_profits(
    returns: np.array, buy_price: float, buy_fee: float
) -> tuple:
    # TODO: add compounding, otherwise this calculation is very misleading
    years_range = range(1, min(21, returns.shape[1] // (52 * 5)))
    monthly_profits, profits_per_trade = [], []
    for years in years_range:
        expected_monthly_profits, expected_profits_per_trade = buy_and_hold(
            returns,
            holding_years=years,
            buy_price=buy_price,
            buy_fee=buy_fee,
            n_simulations=100,
        )
        monthly_profits.append(expected_monthly_profits)
        profits_per_trade.append(expected_profits_per_trade)
    return monthly_profits, profits_per_trade, years_range


def buy_and_hold(
    daily_returns: np.array,  # shape: (n_securities, n_time_steps)
    holding_years: int = 1,
    buy_price: int = 1000,
    buy_fee: float = 0.99,
    n_simulations: int = -1,  # either positive integer or all possible simulations
) -> float:
    """computes expected monthly profits & profits per trade for a naive buy-and-hold strategy via simulations over a holding period"""

    # TODO: include dividends and compounding effects

    holding_period = 52 * 5 * holding_years  # number of weekdays
    daily_profits, profits_per_trade = [], []
    for dr in daily_returns:
        possible_investment_period = range(1, len(dr) - holding_period + 1)
        investment_period = (
            random.choices(possible_investment_period, k=n_simulations)
            if n_simulations > 0
            else possible_investment_period
        )
        for i in investment_period:
            value = buy_price
            for day_idx in range(holding_period):
                value *= dr[i + day_idx]
            daily_profits.append((value - buy_price - buy_fee) / holding_period)
            profits_per_trade.append(value - buy_price - buy_fee)
    expected_monthly_profits = round(np.mean(daily_profits) * 52 * 5 / 12, 2)
    expected_profits_per_trade = round(np.mean(profits_per_trade), 2)

    return expected_monthly_profits, expected_profits_per_trade


def compute_expected_weekly_trading_profits(
    returns: np.array, buy_price: float, buy_fee: float
) -> pd.DataFrame:
    weeks, precisions, monthly_profits, profits_per_trade = [], [], [], []
    for week in range(1, 5):
        for precision in range(0, 101, 5):
            expected_monthly_profits, expected_profits_per_trade = weekly_trading(
                returns,
                holding_weeks=week,
                buy_price=buy_price,
                buy_fee=buy_fee,
                precision=precision / 100,
            )
            weeks.append(week)
            precisions.append(precision / 100)
            monthly_profits.append(expected_monthly_profits)
            profits_per_trade.append(expected_profits_per_trade)
    return pd.DataFrame(
        {
            "Holding Weeks": weeks,
            "Model Precision": precisions,
            "Expected Monthly Profit [€]": monthly_profits,
            "Expected Profit per Trade [€]": profits_per_trade,
        }
    )


def weekly_trading(
    daily_returns: np.array,  # shape: (n_securities, n_time_steps)
    holding_weeks: int = 1,
    buy_price: int = 1000,
    buy_fee: float = 0.99,
    precision: float = 0.5,
    n_simulations: int = -1,  # either positive integer or all possible simulations
) -> float:
    """computes expected monthly profits & profits per trade when buying on a Monday and selling on a Friday, given a positive predictive value (model precision)"""

    holding_period = 5 * holding_weeks
    daily_profits, profits_per_trade = [], []
    for dr in daily_returns:
        return_periods = [
            dr[i : i + holding_period]  # returns from Monday to Friday
            for i in range(0, len(dr), holding_period)
        ]
        return_periods = (
            random.choices(return_periods, k=n_simulations)
            if n_simulations > 0
            else return_periods
        )
        for returns in return_periods:
            value = buy_price
            for ret in returns:
                value *= ret
            rand = random.random()
            if ((value > buy_price) and (rand < precision)) or (
                (value < buy_price) and (rand > precision)
            ):  # true positive or false positive
                daily_profits.append((value - buy_price - buy_fee) / holding_period)
                profits_per_trade.append((value - buy_price - buy_fee))
    expected_monthly_profits = round(np.mean(daily_profits) * 52 * 5 / 12, 2)
    expected_profits_per_trade = round(np.mean(profits_per_trade), 2)

    return expected_monthly_profits, expected_profits_per_trade


def visualize_averaged_yearly_log_returns(log_returns: np.array, start_year: int):
    mean_log_returns = np.mean(log_returns, 0)
    mean_yearly_log_returns = []
    for i in range(0, len(mean_log_returns), 52 * 5):
        mean_yearly_log_returns.append(np.mean(mean_log_returns[i : i + 52 * 5]))
    mean_log_returns = pd.DataFrame(
        {
            "Year": list(range(start_year, start_year + len(mean_yearly_log_returns))),
            "Average Log Return": mean_yearly_log_returns,
        }
    )
    mean_log_returns.set_index("Year", inplace=True)
    sns.barplot(mean_log_returns.reset_index(), x="Year", y="Average Log Return")
    plt.title(f"Yearly Log Returns Averaged over {len(log_returns)} ATX Securities")
    plt.show()


def visualize_expected_buy_and_hold_profits(
    monthly_profits: list, profits_per_trade: list, years_range: range, buy_price: float
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(
        f"Expected Profits when Investing {buy_price}€ in Top ATX Stocks for Different Holding Periods"
    )
    sns.barplot(ax=ax1, x=list(years_range), y=monthly_profits)
    ax1.set_xlabel("Holding Period [Years]")
    ax1.set_ylabel("Expected Monthly Profit [€]")
    sns.barplot(ax=ax2, x=list(years_range), y=profits_per_trade)
    ax2.set_xlabel("Holding Period [Years]")
    ax2.set_ylabel("Expected Profit per Trade [€]")
    plt.show()


def visualize_expected_weekly_trading_profits(
    df: pd.DataFrame, buy_price: float
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(
        f"Expected Profits when Trading Top ATX Stocks Initially Worth {buy_price}€ for Different Model Precisions"
    )
    sns.barplot(
        ax=ax1,
        x="Model Precision",
        y="Expected Monthly Profit [€]",
        hue="Holding Weeks",
        data=df,
    )
    sns.barplot(
        ax=ax2,
        x="Model Precision",
        y="Expected Profit per Trade [€]",
        hue="Holding Weeks",
        data=df,
    )
    ax1.axhline(y=50, c="gray", linewidth=2)
    ax1.axhline(y=100, c="gray", linewidth=1, linestyle="--")
    ax2.axhline(y=50, c="gray", linewidth=2)
    ax2.axhline(y=100, c="gray", linewidth=1, linestyle="--")
    plt.show()
