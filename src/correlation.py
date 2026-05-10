"""Correlation analysis helpers for aligning sentiment with stock returns.

Provides composable functions to normalise dates to trading days, merge
datasets, compute correlation coefficients, and produce summary reports.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# 1.  Date alignment — shift headlines to the next trading day
# ---------------------------------------------------------------------------


def align_to_trading_day(
    dates: pd.Series,
    trading_calendar: pd.DatetimeIndex,
) -> pd.Series:
    """Shift each date forward to the next trading day.

    Dates that are already trading days remain unchanged.
    Weekend/holiday dates are advanced to the next available trading day.

    Parameters
    ----------
    dates : pd.Series
        Input datetime series (may include weekends/holidays).
    trading_calendar : pd.DatetimeIndex
        Sorted index of valid trading days.

    Returns
    -------
    pd.Series
        Aligned datetime series.
    """
    logger.info("Aligning %d dates to trading calendar", len(dates))
    aligned = dates.apply(
        lambda d: trading_calendar[trading_calendar >= d].min()
    )
    return aligned


# ---------------------------------------------------------------------------
# 2.  Aggregate sentiment by trading day
# ---------------------------------------------------------------------------


def aggregate_sentiment(
    df: pd.DataFrame,
    date_col: str = "date_utc",
    score_cols: Optional[List[str]] = None,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Average sentiment scores per trading day, optionally by group.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *date_col*.
    date_col : str
        Datetime column to group by.
    score_cols : list of str or None
        Numeric sentiment columns to aggregate (default: VADER + TextBlob).
    group_col : str or None
        If set, group by this column as well (e.g. ``"stock"``).

    Returns
    -------
    pd.DataFrame
        Grouped means with a ``DatetimeIndex``.
    """
    if score_cols is None:
        score_cols = [c for c in ["sentiment_vader", "sentiment_textblob"] if c in df.columns]

    by = [pd.Grouper(key=date_col, freq="D")]
    if group_col is not None:
        by.append(group_col)

    result = df.groupby(by)[score_cols].mean().dropna(how="all")
    logger.info("Aggregated to %d trading-day rows", len(result))
    return result


# ---------------------------------------------------------------------------
# 3.  Compute daily stock returns
# ---------------------------------------------------------------------------


def daily_returns(close: pd.Series) -> pd.Series:
    """Daily simple returns in percent: ((Ct - Ct-1) / Ct-1) * 100."""
    return close.pct_change() * 100


# ---------------------------------------------------------------------------
# 4.  Merge sentiment with returns on aligned dates
# ---------------------------------------------------------------------------


def merge_sentiment_returns(
    sentiment: pd.DataFrame,
    returns: pd.DataFrame,
    sentiment_date_col: str = "date",
) -> pd.DataFrame:
    """Inner-join daily sentiment averages with daily stock returns.

    Parameters
    ----------
    sentiment : pd.DataFrame
        Sentiment data with a date column.
    returns : pd.DataFrame
        Stock returns with a ``DatetimeIndex``.
    sentiment_date_col : str
        Column in *sentiment* holding the (aligned) date.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with ``DatetimeIndex``.
    """
    # Ensure sentiment has a proper datetime index
    if sentiment_date_col in sentiment.columns:
        sentiment = sentiment.set_index(sentiment_date_col)

    sentiment.index = pd.to_datetime(sentiment.index)
    returns.index = pd.to_datetime(returns.index)

    merged = sentiment.join(returns, how="inner").dropna()
    logger.info("Merged dataset: %d rows", len(merged))
    return merged


# ---------------------------------------------------------------------------
# 5.  Compute Pearson / Spearman correlation
# ---------------------------------------------------------------------------


def compute_correlations(
    df: pd.DataFrame,
    sentiment_cols: List[str],
    target_cols: List[str],
) -> pd.DataFrame:
    """Pearson and Spearman correlation between each sentiment/return pair.

    Returns a DataFrame with columns ``sentiment``, ``target``,
    ``pearson_r``, ``pearson_pval``, ``spearman_rho``, ``spearman_pval``.
    """
    rows = []
    for s in sentiment_cols:
        for t in target_cols:
            clean = df[[s, t]].dropna()
            if len(clean) < 3:
                continue
            p_r, p_p = pearsonr(clean[s], clean[t])
            s_r, s_p = spearmanr(clean[s], clean[t])
            rows.append(
                {
                    "sentiment": s,
                    "target": t,
                    "pearson_r": round(p_r, 4),
                    "pearson_pval": round(p_p, 6),
                    "spearman_rho": round(s_r, 4),
                    "spearman_pval": round(s_p, 6),
                    "n_obs": len(clean),
                }
            )
    result = pd.DataFrame(rows)
    logger.info("Computed %d correlation pairs", len(result))
    return result


# ---------------------------------------------------------------------------
# 6.  Lagged correlation (sentiment leads returns by N days)
# ---------------------------------------------------------------------------


def lagged_correlation(
    df: pd.DataFrame,
    sentiment_col: str = "vader_avg",
    return_col: str = "return",
    max_lag: int = 5,
) -> pd.DataFrame:
    """Correlation between sentiment and *future* returns at various lags.

    A lag of 1 means ``sentiment[t]`` vs ``returns[t+1]``.

    Returns a DataFrame with columns ``lag``, ``pearson_r``, ``pearson_pval``.
    """
    rows = []
    for lag in range(0, max_lag + 1):
        s = df[sentiment_col]
        r = df[return_col].shift(-lag)
        clean = pd.DataFrame({sentiment_col: s, return_col: r}).dropna()
        if len(clean) < 3:
            continue
        p_r, p_p = pearsonr(clean[sentiment_col], clean[return_col])
        rows.append(
            {
                "lag_days": lag,
                "pearson_r": round(p_r, 4),
                "pearson_pval": round(p_p, 6),
                "n_obs": len(clean),
            }
        )
    return pd.DataFrame(rows).sort_values("lag_days")
