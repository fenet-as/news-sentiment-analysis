#!/usr/bin/env python
"""Download sample financial news headlines and stock price data.

Usage
-----
    python scripts/download_data.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import fetch_stock_prices, load_csv  # noqa: E402
from src.utils import data_dir, setup_logger  # noqa: E402

logger = setup_logger("download_data")


def main() -> None:
    logger.info("Downloading stock price data for FAANG tickers...")

    prices = fetch_stock_prices(
        tickers=["META", "AAPL", "AMZN", "NFLX", "GOOG"],
        start="2023-01-01",
    )

    out_path = data_dir("raw") / "stock_prices.csv"
    prices.to_csv(out_path)
    logger.info("Saved stock prices to %s (%d rows)", out_path, len(prices))

    logger.info("Done. Place your headline CSV as data/raw/headlines.csv.")


if __name__ == "__main__":
    main()
