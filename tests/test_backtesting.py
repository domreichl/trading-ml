import numpy as np

from backtesting import (
    compute_expected_buy_and_hold_profits,
    compute_expected_weekly_trading_profits,
    buy_and_hold,
    weekly_trading,
)


def test_compute_expected_buy_and_hold_profits():
    (
        monthly_profits,
        profits_per_trade,
        years_range,
    ) = compute_expected_buy_and_hold_profits(np.ones((3, 1119)), 1000, 2.99)
    assert monthly_profits == [-0.25, -0.12, -0.08]
    assert profits_per_trade == [-2.99, -2.99, -2.99]
    assert years_range == range(1, 4)


def test_compute_expected_weekly_trading_profits():
    df = compute_expected_weekly_trading_profits(np.ones((3, 1119)) + 0.1, 1000, 2.99)
    assert len(df) == 84


def test_buy_and_hold():
    expected_monthly_profits, expected_profits_per_trade = buy_and_hold(
        daily_returns=np.array([[1.01] * 5 * 52 * 2] * 10),
        holding_years=1,
        buy_price=1000,
        buy_fee=10,
        n_simulations=10,
    )
    assert expected_monthly_profits > 1000
    assert expected_profits_per_trade > 10000


def test_weekly_trading():
    expected_monthly_profits, expected_profits_per_trade = weekly_trading(
        daily_returns=np.array([[1.01] * 5 * 52 * 2] * 10),
        holding_weeks=1,
        buy_price=1000,
        buy_fee=10,
        precision=1.0,
        n_simulations=10,
    )
    assert expected_monthly_profits == 132.62
    assert expected_profits_per_trade == 30.6
