"""Data acquisition and loading utilities.

Provides functions to fetch historical stock prices via ``yfinance``
and to load raw CSV/text datasets from the ``data/raw/`` directory.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

from src.utils import data_dir, setup_logger

logger = setup_logger(__name__)


def fetch_stock_prices(
    tickers: List[str],
    start: str = "2023-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download daily adjusted close prices for a list of tickers.

    Parameters
    ----------
    tickers : list of str
        Stock ticker symbols (e.g., ``["AAPL", "GOOG"]``).
    start : str
        Start date in ``YYYY-MM-DD`` format.
    end : str or None
        End date (defaults to today).
    interval : str
        Data interval (``"1d"``, ``"1wk"``, ``"1mo"``).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a ``Date`` index and one column per ticker.
    """
    logger.info("Downloading %s from %s to %s", tickers, start, end or "today")
    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs("Close", axis=1, level=1)
    else:
        close = data[["Close"]] if len(tickers) == 1 else data

    close.columns = [t.upper() for t in close.columns]
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    close = close.ffill().dropna(how="all", axis=1)
    logger.info(
        "Retrieved %s rows x %s columns", close.shape[0], close.shape[1]
    )
    return close


def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from the ``data/raw/`` directory.

    Parameters
    ----------
    filename : str
        File name (e.g., ``"headlines.csv"``).

    Returns
    -------
    pd.DataFrame
    """
    path: Path = data_dir("raw") / filename
    logger.info("Loading CSV from %s", path)
    return pd.read_csv(path, parse_dates=True, infer_datetime_format=True)
