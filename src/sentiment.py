"""Sentiment analysis utilities using VADER and TextBlob.

Provides composable functions to score textual data (news headlines,
tweets, etc.) and return tidy DataFrames with standardised columns.
"""

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

from typing import List, Optional

from src.utils import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy initialisation of resource-heavy components
# ---------------------------------------------------------------------------
_sia: SentimentIntensityAnalyzer | None = None


def _vader() -> SentimentIntensityAnalyzer:
    """Return a singleton VADER analyser, downloading the lexicon once."""
    global _sia
    if _sia is None:
        nltk.download("vader_lexicon", quiet=True)
        _sia = SentimentIntensityAnalyzer()
    return _sia


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------


def vader_score(text: str) -> float:
    """Return the VADER compound sentiment score for *text*.

    Returns a float in ``[-1, 1]`` where negative values indicate
    negative sentiment and positive values indicate positive sentiment.
    """
    return _vader().polarity_scores(text)["compound"]


def textblob_score(text: str) -> float:
    """Return the TextBlob polarity score for *text* (range ``[-1, 1]``)."""
    return TextBlob(text).sentiment.polarity


def score_headlines(
    df: pd.DataFrame,
    text_column: str = "headline",
) -> pd.DataFrame:
    """Augment a DataFrame with VADER and TextBlob sentiment columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *text_column*.
    text_column : str
        Name of the column holding headline text.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with two new columns:
        ``"sentiment_vader"`` and ``"sentiment_textblob"``.
    """
    logger.info("Scoring %d headlines with VADER and TextBlob", len(df))
    result = df.copy()
    result["sentiment_vader"] = result[text_column].apply(vader_score)
    result["sentiment_textblob"] = result[text_column].apply(textblob_score)
    return result


def categorize_sentiment(
    df: pd.DataFrame,
    score_col: str = "sentiment_vader",
    pos_thresh: float = 0.05,
    neg_thresh: float = -0.05,
) -> pd.DataFrame:
    """Add a ``sentiment_category`` column based on score thresholds.

    Categories: ``"positive"``, ``"neutral"``, ``"negative"``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *score_col*.
    score_col : str
        Column with numeric sentiment scores.
    pos_thresh : float
        Scores >= this value are positive.
    neg_thresh : float
        Scores <= this value are negative.
    """
    result = df.copy()
    conditions = [
        result[score_col] >= pos_thresh,
        result[score_col] <= neg_thresh,
    ]
    choices = ["positive", "negative"]
    result["sentiment_category"] = np.select(conditions, choices, default="neutral")
    return result
