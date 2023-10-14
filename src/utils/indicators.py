import numpy as np


def print_market_signals(top_stock: str, overbought: int, bullish: int) -> None:
    print(f"\nThe market for {top_stock} is currently", end=" ")
    if bullish > 0:
        print("bullish and", end=" ")
    elif bullish < 0:
        print("bearish and", end=" ")
    elif bullish == 0:
        print("neutral and", end=" ")
    if overbought > 0:
        if overbought > 1:
            print("very", end=" ")
        print("overbought.")
    elif overbought < 0:
        if overbought < -1:
            print("very", end=" ")
        print("oversold.")
    else:
        print("neither overbought nor oversold.")


def compute_market_signals(prices: np.array) -> tuple[int, int]:
    """
    overbought:
        >0 = overbought: upward trend may reverse
        <0 = oversold: downward trend may reverse
        0 = no indication
    bullish:
        >0 = bullish: signal to open long, sell short
        <0 = bearish: signal to sell long, open short
        0 = neutral
    """
    overbought = 0
    bullish = 0

    fso = fast_stochastic_oscillator(prices)
    if fso > 80:
        overbought += 1
    elif fso < 20:
        overbought -= 1

    bbb = bollinger_band_breakout(prices)
    overbought += bbb

    macdc = moving_average_convergence_divergence_crossover(prices)
    bullish += macdc

    return overbought, bullish


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
