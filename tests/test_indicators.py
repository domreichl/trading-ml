import numpy as np
import pandas as pd
from pathlib import Path
from ta.momentum import StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands

from utils.data_preprocessing import preprocess_data
from utils.indicators import (
    bollinger_band_breakout,
    moving_average_convergence_divergence,
    fast_stochastic_oscillator,
)


mts = preprocess_data(Path(__file__).parent.joinpath("test_data.csv"), test_days=10)
prices = list(mts.close_prices.values())[0]


def test_bollinger_band_breakout():
    breakout, upper, lower = bollinger_band_breakout(prices)
    bb_indicator = BollingerBands(close=pd.Series(prices), window=20, window_dev=2)
    assert round(upper, 6) == round(bb_indicator.bollinger_hband().iloc[-1], 6)
    assert round(lower, 6) == round(bb_indicator.bollinger_lband().iloc[-1], 6)
    assert (
        breakout
        == bb_indicator.bollinger_hband_indicator().iloc[-1]
        == bb_indicator.bollinger_lband_indicator().iloc[-1]
        == 0
    )


def test_moving_average_convergence_divergence():
    _, macd, signal, relative_change = moving_average_convergence_divergence(prices)
    macd_indicator = MACD(pd.Series(prices))
    assert round(macd, 6) == round(macd_indicator.macd().iloc[-1], 6)
    assert round(signal, 6) == round(macd_indicator.macd_signal().iloc[-1], 6)
    assert round((macd - signal), 6) == round(macd_indicator.macd_diff().iloc[-1], 6)
    print(round(relative_change, 4))
    assert round(relative_change, 4) == 0.0001


def test_fast_stochastic_oscillator():
    period = 260
    recent_period = prices[-period:]
    fso = fast_stochastic_oscillator(recent_period, period=period)
    so_indicator = StochasticOscillator(
        pd.Series(np.broadcast_to(max(recent_period), (period))),
        pd.Series(np.broadcast_to(min(recent_period), (period))),
        pd.Series(recent_period),
        window=period,
    )
    assert fso == round(so_indicator.stoch().iloc[-1]) == 1
