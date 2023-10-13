import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
