import numpy as np
import pandas as pd
from ta.trend import ema_indicator


def interpret_market_signals(top_stock: str, trend: int, state: int) -> tuple[str]:
    trend_str = "neutral"
    state_str = "neither overbought nor oversold"
    if trend > 0:
        trend_str = "bullish"
    elif trend < 0:
        trend_str = "bearish"
    if state > 0:
        state_str = "overbought"
    elif state < 0:
        state_str = "oversold"
    if abs(state) > 1:
        state_str = "very " + state_str
    print(f"\nThe market for {top_stock} is currently {trend_str} and {state_str}.")
    return trend_str, state_str


def compute_market_signals(prices: np.array) -> tuple[int, int]:
    trend = 0
    state = 0

    macdc, _, _, _ = moving_average_convergence_divergence(prices)
    trend += macdc

    fso = fast_stochastic_oscillator(prices)
    if fso > 80:
        state += 1
    elif fso < 20:
        state -= 1

    bbb, _, _ = bollinger_band_breakout(prices)
    state += bbb

    return trend, state, macdc, fso, bbb


def moving_average_convergence_divergence(
    prices: np.array, threshold: float = 0.5
) -> tuple[int, float, float]:
    ema26 = ema_indicator(pd.Series(prices), window=26)
    ema12 = ema_indicator(pd.Series(prices), window=12)
    signal = ema_indicator(ema12 - ema26, window=9)
    signal = signal.iloc[-1]
    macd = ema12.iloc[-1] - ema26.iloc[-1]
    relative_change = (macd - signal) / signal
    crossover = 0
    if abs(relative_change) > threshold:
        if macd > signal:
            crossover = 1
        else:
            crossover = -1
    return crossover, macd, signal, relative_change


def fast_stochastic_oscillator(prices: np.array, period: int = 14) -> int:
    recent_period = prices[-period:]
    high = max(recent_period)
    low = min(recent_period)
    current = prices[-1]
    k = (current - low) / (high - low)
    return round(k * 100)


def bollinger_band_breakout(
    prices: np.array, period: int = 20
) -> tuple[int, float, float]:
    recent_period = prices[-period:]
    avg = np.mean(recent_period)
    std = np.std(recent_period)
    upper = avg + 2 * std
    lower = avg - 2 * std
    current = prices[-1]
    breakout = 0
    if current > upper:
        breakout = 1
    elif current < lower:
        breakout = -1
    return breakout, upper, lower
