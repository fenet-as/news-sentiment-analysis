"""Technical indicator calculations built on top of ``pandas``.

All functions accept and return ``pd.Series`` so they compose naturally
with the rest of the data pipeline.
"""

import pandas as pd


def sma(series: pd.Series, window: int = 20) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window).mean()


def ema(series: pd.Series, span: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index ( Wilder's smoothing )."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram.

    Returns a DataFrame with columns ``macd``, ``signal``, ``histogram``.
    """
    ema_fast = ema(series, span=fast)
    ema_slow = ema(series, span=slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, span=signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})


def daily_returns(series: pd.Series) -> pd.Series:
    """Daily simple returns expressed as percentages."""
    return series.pct_change() * 100
