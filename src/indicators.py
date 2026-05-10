"""Technical indicator calculations using pandas, TA-Lib, and pynance.

Every function accepts and returns ``pd.Series`` or ``pd.DataFrame`` for
painless composition with the rest of the pipeline.
"""

from typing import Optional

import numpy as np
import pandas as pd


from src.utils import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# 1.  Data loading  (OHLCV -- CSV or live)
# ---------------------------------------------------------------------------


def load_ohlcv(
    path: Optional[str] = None,
    ticker: str = "AAPL",
    start: str = "2023-01-01",
) -> pd.DataFrame:
    """Load OHLCV data from CSV (preferred) or via ``yfinance``.

    Returns a DataFrame with a ``DatetimeIndex`` sorted ascending.
    """
    if path is not None:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        from src.data_loader import fetch_stock_prices

        closes = fetch_stock_prices([ticker], start=start)
        df = closes.to_frame(ticker)
        df.columns = ["Close"]

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    logger.info("OHLCV loaded -- %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 2.  Moving averages  (TA-Lib primary, pandas fallback)
# ---------------------------------------------------------------------------


def sma(series: pd.Series, window: int = 20) -> pd.Series:
    """Simple Moving Average via TA-Lib."""
    try:
        import talib

        return pd.Series(
            talib.SMA(series.values, timeperiod=window),
            index=series.index,
            name=f"SMA_{window}",
        )
    except Exception:
        return series.rolling(window=window).mean().rename(f"SMA_{window}")


def ema(series: pd.Series, span: int = 20) -> pd.Series:
    """Exponential Moving Average via TA-Lib."""
    try:
        import talib

        return pd.Series(
            talib.EMA(series.values, timeperiod=span),
            index=series.index,
            name=f"EMA_{span}",
        )
    except Exception:
        return series.ewm(span=span, adjust=False).mean().rename(f"EMA_{span}")


# ---------------------------------------------------------------------------
# 3.  RSI  (TA-Lib primary)
# ---------------------------------------------------------------------------


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index via TA-Lib."""
    try:
        import talib

        return pd.Series(
            talib.RSI(series.values, timeperiod=window),
            index=series.index,
            name=f"RSI_{window}",
        )
    except Exception:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        return pd.Series(
            100 - (100 / (1 + rs)), index=series.index, name=f"RSI_{window}"
        )


# ---------------------------------------------------------------------------
# 4.  MACD  (TA-Lib primary)
# ---------------------------------------------------------------------------


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram via TA-Lib.

    Returns a DataFrame with columns ``macd``, ``signal``, ``histogram``.
    """
    try:
        import talib

        m, s, h = talib.MACD(
            series.values, fastperiod=fast, slowperiod=slow, signalperiod=signal
        )
        return pd.DataFrame(
            {"macd": m, "signal": s, "histogram": h}, index=series.index
        )
    except Exception:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        sig_line = macd_line.ewm(span=signal, adjust=False).mean()
        return pd.DataFrame(
            {"macd": macd_line, "signal": sig_line, "histogram": macd_line - sig_line}
        )


# ---------------------------------------------------------------------------
# 5.  Returns & volatility
# ---------------------------------------------------------------------------


def daily_returns(series: pd.Series) -> pd.Series:
    """Daily simple returns in percent."""
    return series.pct_change() * 100


def log_returns(series: pd.Series) -> pd.Series:
    """Continuously compounded (log) returns in percent."""
    return np.log(series / series.shift(1)) * 100


def rolling_volatility(series: pd.Series, window: int = 21) -> pd.Series:
    """Rolling annualised volatility (21-trading-day window)."""
    ret = daily_returns(series).dropna()
    return (ret.rolling(window).std() * np.sqrt(252)).rename("volatility_21d")


# ---------------------------------------------------------------------------
# 6.  Bollinger Bands  (pynance + fallback)
# ---------------------------------------------------------------------------


def bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands via pynance.

    Returns a DataFrame with columns ``upper``, ``mid``, ``lower``.
    """
    try:
        bb = pn.tech.bollinger(series.values, window=window, numsd=num_std)
        return pd.DataFrame(
            {"upper": bb[:, 0], "mid": bb[:, 1], "lower": bb[:, 2]},
            index=series.index,
        )
    except Exception:
        mid = sma(series, window)
        std = series.rolling(window).std()
        return pd.DataFrame(
            {"upper": mid + num_std * std, "mid": mid, "lower": mid - num_std * std},
            index=series.index,
        )


# ---------------------------------------------------------------------------
# 7.  Growth metrics  (pynance)
# ---------------------------------------------------------------------------


def growth_metrics(series: pd.Series, col_name: str = "Close") -> pd.DataFrame:
    """Growth, log-growth, and growth volatility via pynance.

    Parameters
    ----------
    series : pd.Series
        Price series (will be wrapped in a DataFrame for pynance).
    col_name : str
        Column name pynance should look up.
    """
    frame = series.to_frame(col_name)
    gm = pn.tech.growth(frame, selection=col_name)
    lgm = pn.tech.ln_growth(frame, selection=col_name)
    gv = pn.tech.growth_volatility(frame, selection=col_name)
    return pd.DataFrame(
        {
            "growth": gm.iloc[:, 0].values,
            "ln_growth": lgm.iloc[:, 0].values,
            "growth_vol": gv.iloc[:, 0].values,
        },
        index=frame.index[1:],
    )


# ---------------------------------------------------------------------------
# 8.  Convenience:  compute all indicators at once
# ---------------------------------------------------------------------------


def compute_all(df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
    """Compute a broad set of indicators and return a single DataFrame."""
    result = df.copy()
    close = result[close_col]
    result["SMA_20"] = sma(close, 20)
    result["SMA_50"] = sma(close, 50)
    result["EMA_20"] = ema(close, 20)
    result["RSI_14"] = rsi(close, 14)
    macd_df = macd(close)
    for col in macd_df.columns:
        result[f"MACD_{col}"] = macd_df[col]
    bb = bollinger_bands(close)
    for col in bb.columns:
        result[f"BB_{col}"] = bb[col]
    result["daily_return"] = daily_returns(close)
    result["log_return"] = log_returns(close)
    result["volatility_21d"] = rolling_volatility(close, 21)
    logger.info("Computed all indicators -- %d columns", result.shape[1])
    return result
