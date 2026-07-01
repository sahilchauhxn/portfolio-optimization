"""
visualization.py
=================
All plotting utilities for the portfolio optimization project. Every
function saves a PNG to the outputs/ directory and also returns the
matplotlib Figure object (useful in notebooks).
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

if "seaborn-v0_8-whitegrid" in plt.style.available:
    plt.style.use("seaborn-v0_8-whitegrid")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_GMV = "#1f77b4"
COLOR_EW = "#ff7f0e"
COLOR_MSR = "#2ca02c"
COLOR_FRONTIER = "#7f7f7f"


def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    gmv_point: tuple,
    ew_point: tuple,
    msr_point: Optional[tuple] = None,
    individual_assets: Optional[pd.DataFrame] = None,
    filename: str = "efficient_frontier.png",
) -> plt.Figure:
    """
    Plot the Markowitz efficient frontier with the GMV, Equal-Weight, and
    (optionally) Max-Sharpe portfolios highlighted, plus individual
    industry risk/return points as a backdrop for context.

    Parameters
    ----------
    frontier_df : DataFrame with columns ['volatility', 'target_return']
    gmv_point, ew_point, msr_point : tuple (volatility, return)
    individual_assets : DataFrame with columns ['volatility', 'return'] (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(
        frontier_df["volatility"], frontier_df["target_return"],
        color=COLOR_FRONTIER, linewidth=2.2, label="Efficient Frontier", zorder=2,
    )

    if individual_assets is not None:
        ax.scatter(
            individual_assets["volatility"], individual_assets["return"],
            color="lightgray", edgecolor="dimgray", s=40, alpha=0.8,
            label="Individual Industries", zorder=1,
        )

    ax.scatter(*gmv_point, color=COLOR_GMV, s=220, marker="*", edgecolor="black",
               linewidth=0.8, label="Global Minimum Variance (GMV)", zorder=4)
    ax.scatter(*ew_point, color=COLOR_EW, s=140, marker="D", edgecolor="black",
               linewidth=0.8, label="Equal Weight (1/N)", zorder=4)
    if msr_point is not None:
        ax.scatter(*msr_point, color=COLOR_MSR, s=180, marker="^", edgecolor="black",
                   linewidth=0.8, label="Max Sharpe Ratio", zorder=4)

    ax.set_xlabel("Annualized Volatility (Risk)", fontsize=12)
    ax.set_ylabel("Annualized Expected Return", fontsize=12)
    ax.set_title("Efficient Frontier — 30 Industry Portfolios", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="best", frameon=True, fontsize=10)
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_portfolio_allocation(
    weights_dict: dict[str, pd.Series],
    filename: str = "portfolio_allocation.png",
    top_n: int = 30,
) -> plt.Figure:
    """
    Grouped horizontal bar chart comparing portfolio weights (e.g. GMV vs
    Equal Weight) across all 30 industries, sorted by GMV weight.
    """
    weights_df = pd.DataFrame(weights_dict)
    sort_col = list(weights_dict.keys())[0]
    weights_df = weights_df.sort_values(sort_col, ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 10))
    n_portfolios = len(weights_dict)
    bar_height = 0.8 / n_portfolios
    y_pos = np.arange(len(weights_df))

    colors = [COLOR_GMV, COLOR_EW, COLOR_MSR]
    for i, col in enumerate(weights_df.columns):
        ax.barh(
            y_pos + i * bar_height, weights_df[col], height=bar_height,
            label=col, color=colors[i % len(colors)], edgecolor="black", linewidth=0.3,
        )

    ax.set_yticks(y_pos + bar_height * (n_portfolios - 1) / 2)
    ax.set_yticklabels(weights_df.index)
    ax.set_xlabel("Portfolio Weight", fontsize=12)
    ax.set_title("Portfolio Allocation by Industry", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    filename: str = "correlation_heatmap.png",
) -> plt.Figure:
    """
    Heatmap of the pairwise correlation matrix across the 30 industries.
    Low/negative correlations (cooler colors) are exactly what the GMV
    optimizer exploits for diversification benefit.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Correlation Coefficient", fontsize=11)

    ax.set_title("Correlation Matrix — 30 Industry Portfolios", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_cumulative_returns(
    cum_returns_dict: dict[str, pd.Series],
    filename: str = "cumulative_returns.png",
) -> plt.Figure:
    """
    Growth-of-$1 chart comparing the cumulative wealth trajectories of
    multiple portfolios (e.g. GMV vs Equal Weight) over the full sample.
    """
    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = [COLOR_GMV, COLOR_EW, COLOR_MSR]
    for i, (name, series) in enumerate(cum_returns_dict.items()):
        ax.plot(series.index, series.values, label=name, linewidth=2.0, color=colors[i % len(colors)])

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Growth of $1", fontsize=12)
    ax.set_title("Cumulative Portfolio Returns", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.set_yscale("log")
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_rolling_volatility(
    rolling_vol_dict: dict[str, pd.Series],
    filename: str = "rolling_volatility.png",
    window: int = 12,
) -> plt.Figure:
    """
    Rolling `window`-month annualized volatility for each portfolio,
    illustrating how risk concentration (or lack thereof) plays out through
    different market regimes (e.g. GFC 2008, COVID 2020).
    """
    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = [COLOR_GMV, COLOR_EW, COLOR_MSR]
    for i, (name, series) in enumerate(rolling_vol_dict.items()):
        ax.plot(series.index, series.values, label=name, linewidth=1.8, color=colors[i % len(colors)])

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(f"{window}-Month Rolling Annualized Volatility", fontsize=12)
    ax.set_title("Rolling Portfolio Volatility", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    fig.tight_layout()
    _save(fig, filename)
    return fig
