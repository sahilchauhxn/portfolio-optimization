"""
optimizer.py
============
Portfolio optimization routines built on top of Modern Portfolio Theory
(Markowitz, 1952). Implements:

    1. The Global Minimum Variance (GMV) portfolio
    2. The Equal-Weight (1/N) benchmark portfolio
    3. The Maximum Sharpe Ratio (tangency) portfolio
    4. The full Efficient Frontier via target-return-constrained variance
       minimization

All optimizations are solved with scipy.optimize.minimize using
Sequential Least Squares Programming (SLSQP), which natively supports the
equality/inequality constraints we need (budget constraint, no-short-sales
bounds, target-return constraint).

--------------------------------------------------------------------------
MATHEMATICAL BACKGROUND
--------------------------------------------------------------------------
For a portfolio with weight vector w = (w_1, ..., w_N)', expected return
vector mu, and covariance matrix Sigma:

    Portfolio expected return:   E[R_p] = w' mu
    Portfolio variance:          Var(R_p) = w' Sigma w
    Portfolio volatility:        sigma_p = sqrt(w' Sigma w)

GLOBAL MINIMUM VARIANCE (GMV) PORTFOLIO
----------------------------------------
The GMV portfolio solves:

    minimize_w   w' Sigma w
    subject to   sum(w_i) = 1          (fully invested / budget constraint)
                 w_i >= 0  (optional)  (long-only constraint)

This is the unique point on the efficient frontier with the lowest possible
variance, REGARDLESS of expected returns -- it depends ONLY on Sigma. This
makes it attractive in practice because covariance matrices are estimated
far more reliably than expected returns (a well-known result, e.g.
Merton 1980 / Jagannathan & Ma 2003), so GMV portfolios tend to be more
robust out-of-sample than mean-variance-optimal portfolios that rely on
noisy return forecasts.

Closed-form solution (unconstrained, i.e. shorting allowed):
    w* = (Sigma^-1 * 1) / (1' * Sigma^-1 * 1)

where 1 is a vector of ones. We verify our numerical (SLSQP) solution
against this closed-form solution as a sanity check when short-selling is
allowed.

MAXIMUM SHARPE RATIO (TANGENCY) PORTFOLIO
-------------------------------------------
    maximize_w   (w' mu - r_f) / sqrt(w' Sigma w)
    subject to   sum(w_i) = 1,  w_i >= 0 (optional)

EFFICIENT FRONTIER
-------------------
For a grid of target returns mu_target, solve:
    minimize_w   w' Sigma w
    subject to   sum(w_i) = 1
                 w' mu = mu_target
                 w_i >= 0 (optional)
Tracing out (sigma_p, mu_target) pairs gives the efficient frontier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class OptimizationResult:
    """Container for a single portfolio optimization result."""
    weights: pd.Series
    expected_return: float          # annualized
    volatility: float                # annualized
    sharpe_ratio: float
    method: str
    success: bool = True
    message: str = ""


def _portfolio_variance(w: np.ndarray, cov: np.ndarray) -> float:
    """w' Sigma w -- the core quadratic form of Markowitz theory."""
    return w.T @ cov @ w


def _portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    """w' mu -- portfolio expected return."""
    return w @ mu


def gmv_closed_form(cov: pd.DataFrame) -> pd.Series:
    """
    Closed-form (unconstrained, shorting allowed) GMV solution:

        w* = (Sigma^-1 1) / (1' Sigma^-1 1)

    Used purely as an analytical sanity-check against the numerical SLSQP
    solver when no long-only constraint is imposed.
    """
    sigma_inv = np.linalg.inv(cov.values)
    ones = np.ones(len(cov))
    w = sigma_inv @ ones / (ones @ sigma_inv @ ones)
    return pd.Series(w, index=cov.index)


def solve_gmv(
    cov: pd.DataFrame,
    mu: pd.Series,
    long_only: bool = True,
    weight_bounds: Optional[tuple] = None,
) -> OptimizationResult:
    """
    Solve the Global Minimum Variance portfolio via SLSQP.

    Parameters
    ----------
    cov : pd.DataFrame
        Annualized covariance matrix (N x N).
    mu : pd.Series
        Annualized expected returns (N,), used only for reporting the
        resulting portfolio's expected return -- NOT part of the objective.
    long_only : bool
        If True, restrict weights to [0, 1] (no short-selling), which is
        the realistic constraint for most long-only industry allocation
        mandates. If False, weights are only bounded by `weight_bounds`
        (default: unbounded, i.e. shorting allowed).
    weight_bounds : tuple, optional
        Override the per-asset bound, e.g. (-0.2, 0.4) to allow limited
        shorting. Ignored if long_only=True (uses (0, 1) instead).

    Returns
    -------
    OptimizationResult
    """
    n = len(cov)
    cov_arr = cov.values
    mu_arr = mu.values

    if long_only:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    elif weight_bounds is not None:
        bounds = tuple(weight_bounds for _ in range(n))
    else:
        bounds = tuple((-1.0, 1.0) for _ in range(n))  # loosely bounded for numerical stability

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # budget constraint: sum(w) = 1
    ]

    w0 = np.repeat(1.0 / n, n)  # start from equal weight

    result = minimize(
        fun=_portfolio_variance,
        x0=w0,
        args=(cov_arr,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    w_opt = pd.Series(result.x, index=cov.index).clip(lower=0 if long_only else None)
    w_opt = w_opt / w_opt.sum()  # re-normalize to guard against tiny numerical drift

    port_return = float(_portfolio_return(w_opt.values, mu_arr))
    port_vol = float(np.sqrt(_portfolio_variance(w_opt.values, cov_arr)))
    sharpe = port_return / port_vol if port_vol > 0 else np.nan

    return OptimizationResult(
        weights=w_opt,
        expected_return=port_return,
        volatility=port_vol,
        sharpe_ratio=sharpe,
        method="Global Minimum Variance (SLSQP)",
        success=result.success,
        message=result.message,
    )


def solve_equal_weight(cov: pd.DataFrame, mu: pd.Series) -> OptimizationResult:
    """
    The naive 1/N benchmark portfolio: w_i = 1/N for all i.

    DeMiguel, Garlappi & Uppal (2009) famously showed that 1/N is a
    surprisingly hard benchmark to beat out-of-sample, which is exactly
    why we compare GMV against it here.
    """
    n = len(cov)
    w = pd.Series(np.repeat(1.0 / n, n), index=cov.index)
    port_return = float(_portfolio_return(w.values, mu.values))
    port_vol = float(np.sqrt(_portfolio_variance(w.values, cov.values)))
    sharpe = port_return / port_vol if port_vol > 0 else np.nan
    return OptimizationResult(
        weights=w,
        expected_return=port_return,
        volatility=port_vol,
        sharpe_ratio=sharpe,
        method="Equal Weight (1/N)",
    )


def solve_max_sharpe(
    cov: pd.DataFrame,
    mu: pd.Series,
    risk_free_rate: float = 0.0,
    long_only: bool = True,
) -> OptimizationResult:
    """
    Solve for the Maximum Sharpe Ratio (tangency) portfolio:

        maximize_w  (w'mu - r_f) / sqrt(w'Sigma w)

    scipy.optimize.minimize only minimizes, so we minimize the NEGATIVE
    Sharpe ratio instead. This portfolio, combined with the risk-free
    asset, forms the Capital Market Line (CML) in classical MPT.
    """
    n = len(cov)
    cov_arr = cov.values
    mu_arr = mu.values

    def neg_sharpe(w):
        ret = _portfolio_return(w, mu_arr)
        vol = np.sqrt(_portfolio_variance(w, cov_arr))
        return -(ret - risk_free_rate) / vol

    bounds = tuple((0.0, 1.0) for _ in range(n)) if long_only else tuple((-1.0, 1.0) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    w0 = np.repeat(1.0 / n, n)

    result = minimize(
        fun=neg_sharpe, x0=w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    w_opt = pd.Series(result.x, index=cov.index).clip(lower=0 if long_only else None)
    w_opt = w_opt / w_opt.sum()

    port_return = float(_portfolio_return(w_opt.values, mu_arr))
    port_vol = float(np.sqrt(_portfolio_variance(w_opt.values, cov_arr)))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else np.nan

    return OptimizationResult(
        weights=w_opt,
        expected_return=port_return,
        volatility=port_vol,
        sharpe_ratio=sharpe,
        method="Maximum Sharpe Ratio (Tangency)",
        success=result.success,
        message=result.message,
    )


def compute_efficient_frontier(
    cov: pd.DataFrame,
    mu: pd.Series,
    n_points: int = 50,
    long_only: bool = True,
) -> pd.DataFrame:
    """
    Trace the efficient frontier by solving, for a grid of target returns
    spanning [min(mu), max(mu)]:

        minimize_w   w' Sigma w
        subject to   sum(w) = 1
                     w' mu = mu_target
                     w_i >= 0 (if long_only)

    Returns
    -------
    pd.DataFrame with columns ['target_return', 'volatility', 'sharpe']
    """
    n = len(cov)
    cov_arr = cov.values
    mu_arr = mu.values

    target_returns = np.linspace(mu.min(), mu.max(), n_points)
    bounds = tuple((0.0, 1.0) for _ in range(n)) if long_only else tuple((-1.0, 1.0) for _ in range(n))

    frontier_records = []
    w0 = np.repeat(1.0 / n, n)

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: _portfolio_return(w, mu_arr) - t},
        ]
        result = minimize(
            fun=_portfolio_variance, x0=w0, args=(cov_arr,), method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if result.success:
            vol = float(np.sqrt(_portfolio_variance(result.x, cov_arr)))
            frontier_records.append({
                "target_return": target,
                "volatility": vol,
                "sharpe": target / vol if vol > 0 else np.nan,
            })

    return pd.DataFrame(frontier_records)
