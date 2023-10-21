import numpy as np


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

    macdc = moving_average_convergence_divergence_crossover(prices)
    trend += macdc

    fso = fast_stochastic_oscillator(prices)
    if fso > 80:
        state += 1
    elif fso < 20:
        state -= 1

    bbb = bollinger_band_breakout(prices)
    state += bbb

    return trend, state, macdc, fso, bbb


def moving_average_convergence_divergence_crossover(
    prices: np.array, threshold: float = 0.5
) -> int:
    p1, p2, p3 = 26, 12, 9
    ema26 = exponential_moving_average(prices, period=p1)
    ema12 = exponential_moving_average(prices, period=p2)
    signal = exponential_moving_average(ema12[-p3:] - ema26[-p3:], period=p3)
    signal = signal[-1]
    macd = ema12[-1] - ema26[-1]
    relative_change = (macd - signal) / signal
    crossover = 0
    if abs(relative_change) > threshold:
        if macd > signal:
            crossover = 1
        else:
            crossover = -1
    return crossover


def exponential_moving_average(prices: np.array, period: int) -> np.array:
    recent_period = prices[-period:]
    alpha = 2 / (len(recent_period) + 1)
    ema = [recent_period[0]]
    for i in range(1, len(recent_period)):
        ema.append(alpha * recent_period[i] + (1 - alpha) * ema[-1])
    return np.array(ema)


def fast_stochastic_oscillator(prices: np.array, period: int = 14) -> int:
    recent_period = prices[-period:]
    high = max(recent_period)
    low = min(recent_period)
    current = prices[-1]
    k = (current - low) / (high - low)
    return round(k * 100)


def bollinger_band_breakout(prices: np.array, period: int = 20) -> int:
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
    return breakout
