"""Publication-quality visualisation helpers.

Wraps ``matplotlib`` and ``seaborn`` with sensible defaults for
financial time-series and sentiment analysis plots.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

from src.utils import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Global style configuration
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def time_series(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Time Series",
    ylabel: str = "Value",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Plot one or more time-series columns from *df*."""
    if columns is None:
        columns = df.select_dtypes("number").columns.tolist()

    fig, ax = plt.subplots(figsize=figsize)
    for col in columns:
        ax.plot(df.index, df[col], label=col, linewidth=1.2)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()

    ax.set(title=title, ylabel=ylabel, xlabel="")
    ax.legend()
    sns.despine()
    return fig


def sentiment_vs_returns(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment_vader",
    return_col: str = "daily_return",
    title: str = "Sentiment vs. Daily Returns",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Dual-axis plot overlaying sentiment and stock returns."""
    fig, ax1 = plt.subplots(figsize=figsize)

    color1 = sns.color_palette("muted")[0]
    color2 = sns.color_palette("muted")[1]

    ax1.plot(df.index, df[sentiment_col], color=color1, label=sentiment_col, linewidth=1.2)
    ax1.set_ylabel("Sentiment Score", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(df.index, df[return_col], color=color2, label=return_col, linewidth=0.8, alpha=0.7)
    ax2.set_ylabel("Daily Return (%)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set(title=title, xlabel="")
    fig.autofmt_xdate()
    sns.despine()
    return fig


def bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str = "Bar Chart",
    xlabel: str = "",
    ylabel: str = "Count",
    figsize: tuple = (10, 5),
    top_n: Optional[int] = None,
    palette: str = "muted",
) -> plt.Figure:
    """Horizontal bar chart — useful for top‑K frequencies."""
    if top_n is not None:
        data = data.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=data, x=y, y=x, palette=palette, ax=ax)
    ax.set(title=title, xlabel=ylabel, ylabel=xlabel)
    sns.despine()
    return fig


def histogram(
    series: pd.Series,
    title: str = "Distribution",
    xlabel: str = "Value",
    figsize: tuple = (8, 4),
    bins: int = 50,
    kde: bool = True,
    color: str = "steelblue",
) -> plt.Figure:
    """Histogram with optional KDE overlay."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(series, bins=bins, kde=kde, color=color, ax=ax)
    ax.set(title=title, xlabel=xlabel, ylabel="Frequency")
    sns.despine()
    return fig


def price_volume(
    df: pd.DataFrame,
    price_col: str = "Close",
    volume_col: str = "Volume",
    title: str = "Price and Volume",
    figsize: tuple = (14, 7),
) -> plt.Figure:
    """Dual-axis chart: price (line) on left, volume (bars) on right.

    Requires *df* to have a ``DatetimeIndex``.
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    color_price = "black"
    color_vol = sns.color_palette("Blues", 3)[1]

    ax1.plot(df.index, df[price_col], color=color_price, linewidth=1.5)
    ax1.set(ylabel="Price (USD)", title=title)

    ax2 = ax1.twinx()
    ax2.bar(df.index, df[volume_col], width=1.5, color=color_vol, alpha=0.4, label="Volume")
    ax2.set(ylabel="Volume")

    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines2:
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    sns.despine()
    return fig


def correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Plot a heatmap of the correlation matrix for numeric columns."""
    corr = df.select_dtypes("number").corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set(title=title)
    return fig
