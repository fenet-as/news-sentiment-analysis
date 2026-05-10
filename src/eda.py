"""Reusable exploratory data analysis (EDA) helpers for financial news.

Provides composable functions that return DataFrames or dicts suitable
for downstream plotting and reporting.  Every function accepts a
``DataFrame`` and returns transformed data so callers stay in control
of visualisation.
"""

import re
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.utils import setup_logger

logger = setup_logger(__name__)

_WORD_PATTERN = re.compile(r"\b[a-zA-Z]{3,}\b")


# ---------------------------------------------------------------------------
# 1.  Data loading and basic inspection
# ---------------------------------------------------------------------------


def load_headlines(
    path: str = "data/raw/headlines.csv",
    parse_dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load the headlines CSV from *path* (relative to project root).

    Parameters
    ----------
    path : str
        Relative or absolute path to the CSV file.
    parse_dates : list of str or None
        Columns to attempt date parsing on.

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Loading headlines from %s", path)
    df = pd.read_csv(path, parse_dates=parse_dates)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Return a summary dict with shape, column info, and memory usage."""
    logger.info("Inspecting dataset — shape %s", df.shape)
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_bytes": df.memory_usage(deep=True).sum(),
        "head": df.head(3),
        "tail": df.tail(3),
    }


# ---------------------------------------------------------------------------
# 2.  Missing value analysis
# ---------------------------------------------------------------------------


def analyze_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with null count and percentage per column."""
    logger.info("Computing missing-value summary")
    missing = pd.DataFrame({
        "column": df.columns,
        "null_count": df.isnull().sum().values,
        "null_pct": (df.isnull().sum() / len(df) * 100).values,
        "dtype": df.dtypes.astype(str).values,
    })
    missing = missing.sort_values("null_count", ascending=False).reset_index(drop=True)
    return missing


# ---------------------------------------------------------------------------
# 3.  Datetime parsing and timezone handling
# ---------------------------------------------------------------------------


def normalize_dates(
    df: pd.DataFrame,
    col: str = "date",
    utc: bool = True,
) -> pd.DataFrame:
    """Parse *col* as datetime, localise to UTC, and extract temporal features.

    Adds columns:
        ``date_utc``, ``year``, ``month``, ``day``, ``dayofweek``, ``hour``
    """
    logger.info("Parsing datetime column '%s'", col)
    result = df.copy()

    raw = result[col]
    parsed = pd.to_datetime(raw, utc=utc, errors="coerce")

    result["date_utc"] = parsed
    result["year"] = parsed.dt.year
    result["month"] = parsed.dt.month
    result["day"] = parsed.dt.day
    result["dayofweek"] = parsed.dt.dayofweek
    result["hour"] = parsed.dt.hour

    n_coerced = parsed.isna().sum()
    if n_coerced:
        logger.warning(
            "%d rows could not be parsed and were set to NaT", n_coerced
        )

    return result


# ---------------------------------------------------------------------------
# 4.  Headline length analysis
# ---------------------------------------------------------------------------


def compute_headline_lengths(
    df: pd.DataFrame,
    text_col: str = "headline",
) -> pd.DataFrame:
    """Add ``char_len`` and ``word_count`` columns to a copy of *df*.

    ``word_count`` counts tokens of at least 3 alphabetic characters.
    """
    result = df.copy()
    result["char_len"] = result[text_col].fillna("").apply(len)
    result["word_count"] = result[text_col].fillna("").apply(
        lambda t: len(_WORD_PATTERN.findall(t))
    )
    logger.info(
        "Headline char_len: mean=%.1f  median=%.1f  max=%d",
        result["char_len"].mean(),
        result["char_len"].median(),
        result["char_len"].max(),
    )
    return result


# ---------------------------------------------------------------------------
# 5.  Publisher frequency
# ---------------------------------------------------------------------------


def publisher_frequency(
    df: pd.DataFrame,
    col: str = "publisher",
    top_n: int = 20,
) -> pd.DataFrame:
    """Return top *top_n* publishers by article count."""
    counts = df[col].value_counts().head(top_n).reset_index()
    counts.columns = ["publisher", "article_count"]
    return counts


# ---------------------------------------------------------------------------
# 6.  Time-series publication frequency
# ---------------------------------------------------------------------------


def publication_frequency(
    df: pd.DataFrame,
    date_col: str = "date_utc",
    freq: str = "D",
) -> pd.Series:
    """Resample article counts by *freq* (e.g. ``'D'``, ``'W'``, ``'M'``)."""
    series = df.set_index(date_col).index
    return pd.Series(
        index=series, data=1, name="count"
    ).resample(freq).sum()


# ---------------------------------------------------------------------------
# 7.  Keyword extraction
# ---------------------------------------------------------------------------


def extract_keywords_cv(
    df: pd.DataFrame,
    text_col: str = "headline",
    ngram_range: Tuple[int, int] = (1, 2),
    top_n: int = 20,
    stop_words: Optional[List[str]] = None,
    min_df: int = 3,
) -> pd.DataFrame:
    """Extract top *top_n* n-grams via ``CountVectorizer``.

    Returns a DataFrame with columns ``keyword`` and ``count``.
    """
    logger.info("Extracting top %d keywords with CountVectorizer", top_n)
    vec = CountVectorizer(
        ngram_range=ngram_range,
        stop_words=stop_words or "english",
        min_df=min_df,
    )
    matrix = vec.fit_transform(df[text_col].fillna(""))
    counts = matrix.sum(axis=0).A1
    keywords = vec.get_feature_names_out()

    top_idx = counts.argsort()[::-1][:top_n]
    return pd.DataFrame({
        "keyword": keywords[top_idx],
        "count": counts[top_idx].astype(int),
    }).reset_index(drop=True)


def extract_keywords_tfidf(
    df: pd.DataFrame,
    text_col: str = "headline",
    ngram_range: Tuple[int, int] = (1, 2),
    top_n: int = 20,
    stop_words: Optional[List[str]] = None,
    min_df: int = 3,
) -> pd.DataFrame:
    """Extract top *top_n* terms via TF-IDF (mean score across documents).

    Returns a DataFrame with columns ``keyword`` and ``tfidf_score``.
    """
    logger.info("Extracting top %d keywords with TF-IDF", top_n)
    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words=stop_words or "english",
        min_df=min_df,
    )
    matrix = vec.fit_transform(df[text_col].fillna(""))
    mean_scores = matrix.mean(axis=0).A1

    top_idx = mean_scores.argsort()[::-1][:top_n]
    return pd.DataFrame({
        "keyword": vec.get_feature_names_out()[top_idx],
        "tfidf_score": mean_scores[top_idx].round(4),
    }).reset_index(drop=True)
