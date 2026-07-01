"""
main.py
=======
End-to-end pipeline for:
    "Portfolio Optimization using Kenneth French 30 Industry Portfolios"

Run with:
    python main.py

This script:
    1. Loads (downloads if possible, else falls back to synthetic) monthly
       returns for the Ken French 30 Industry Portfolios.
    2. Computes expected returns and the annualized covariance matrix.
    3. Solves the Global Minimum Variance (GMV) portfolio via
       scipy.optimize.minimize (SLSQP), long-only.
    4. Benchmarks GMV against the Equal-Weight (1/N) portfolio, and also
       reports the Max Sharpe Ratio portfolio for context.
    5. Computes performance metrics: annualized return, volatility, Sharpe
       ratio, max drawdown, correlation matrix.
    6. Produces all required visualizations into outputs/.
    7. Prints a clean summary report to stdout and saves it as a CSV.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import pandas as pd

from src import data_loader, optimizer, metrics, visualization


def main():
    print("=" * 78)
    print("PORTFOLIO OPTIMIZATION — KENNETH FRENCH 30 INDUSTRY PORTFOLIOS")
    print("Global Minimum Variance vs. Equal Weight vs. Max Sharpe")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data...")
    returns_df = data_loader.load_industry_returns()
    print(f"      Loaded {returns_df.shape[0]} months x {returns_df.shape[1]} industries "
          f"({returns_df.index.min().date()} to {returns_df.index.max().date()})")

    # ------------------------------------------------------------------
    # 2. EXPECTED RETURNS & COVARIANCE
    # ------------------------------------------------------------------
    print("\n[2/6] Computing expected returns and covariance matrix...")
    mu = data_loader.compute_expected_returns(returns_df, annualize=True)
    cov = data_loader.compute_covariance_matrix(returns_df, annualize=True)
    print(f"      Expected annual return range: [{mu.min():.2%}, {mu.max():.2%}]")
    print(f"      Annualized volatility range:  "
          f"[{np.sqrt(np.diag(cov)).min():.2%}, {np.sqrt(np.diag(cov)).max():.2%}]")

    # ------------------------------------------------------------------
    # 3. OPTIMIZE PORTFOLIOS
    # ------------------------------------------------------------------
    print("\n[3/6] Solving portfolio optimizations...")
    gmv = optimizer.solve_gmv(cov, mu, long_only=True)
    ew = optimizer.solve_equal_weight(cov, mu)
    msr = optimizer.solve_max_sharpe(cov, mu, risk_free_rate=0.0, long_only=True)

    # Sanity check GMV numerical solution against closed-form (shorting-allowed) solution
    gmv_unconstrained = optimizer.solve_gmv(cov, mu, long_only=False)
    gmv_closed_form = optimizer.gmv_closed_form(cov)
    closed_form_vol = float(np.sqrt(gmv_closed_form.values @ cov.values @ gmv_closed_form.values))
    print(f"      GMV (long-only, SLSQP)    | success={gmv.success:<5} | vol={gmv.volatility:.4%}")
    print(f"      GMV (unconstrained, SLSQP)| success={gmv_unconstrained.success:<5} | "
          f"vol={gmv_unconstrained.volatility:.4%}")
    print(f"      GMV (closed-form check)   | vol={closed_form_vol:.4%}  "
          f"<- should match unconstrained SLSQP result")
    print(f"      Equal Weight               | vol={ew.volatility:.4%}")
    print(f"      Max Sharpe Ratio            | success={msr.success:<5} | vol={msr.volatility:.4%}")

    weights_dict = {
        "GMV": gmv.weights,
        "Equal Weight": ew.weights,
        "Max Sharpe": msr.weights,
    }

    # ------------------------------------------------------------------
    # 4. PERFORMANCE METRICS
    # ------------------------------------------------------------------
    print("\n[4/6] Computing performance metrics on realized history...")
    summary = metrics.summary_table(returns_df, weights_dict)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n" + summary.to_string())

    corr = metrics.correlation_matrix(returns_df)
    avg_corr = (corr.values.sum() - len(corr)) / (len(corr) ** 2 - len(corr))
    print(f"\n      Average pairwise industry correlation: {avg_corr:.3f}")

    # Save summary + weights + correlation to outputs/
    os.makedirs("outputs", exist_ok=True)
    summary.to_csv("outputs/performance_summary.csv")
    pd.DataFrame(weights_dict).to_csv("outputs/portfolio_weights.csv")
    corr.to_csv("outputs/correlation_matrix.csv")
    print("\n      Saved: outputs/performance_summary.csv, "
          "outputs/portfolio_weights.csv, outputs/correlation_matrix.csv")

    # ------------------------------------------------------------------
    # 5. EFFICIENT FRONTIER
    # ------------------------------------------------------------------
    print("\n[5/6] Computing the efficient frontier (this may take a few seconds)...")
    frontier = optimizer.compute_efficient_frontier(cov, mu, n_points=40, long_only=True)

    # ------------------------------------------------------------------
    # 6. VISUALIZATIONS
    # ------------------------------------------------------------------
    print("\n[6/6] Generating visualizations into outputs/ ...")

    individual_assets = pd.DataFrame({
        "volatility": np.sqrt(np.diag(cov)),
        "return": mu.values,
    }, index=cov.index)

    visualization.plot_efficient_frontier(
        frontier_df=frontier,
        gmv_point=(gmv.volatility, gmv.expected_return),
        ew_point=(ew.volatility, ew.expected_return),
        msr_point=(msr.volatility, msr.expected_return),
        individual_assets=individual_assets,
    )
    visualization.plot_portfolio_allocation(weights_dict)
    visualization.plot_correlation_heatmap(corr)

    cum_returns_dict = {
        name: metrics.cumulative_returns(metrics.portfolio_monthly_returns(returns_df, w))
        for name, w in weights_dict.items()
    }
    visualization.plot_cumulative_returns(cum_returns_dict)

    rolling_vol_dict = {
        name: metrics.rolling_volatility(metrics.portfolio_monthly_returns(returns_df, w))
        for name, w in weights_dict.items()
    }
    visualization.plot_rolling_volatility(rolling_vol_dict)

    print("      Saved 5 charts to outputs/:")
    print("        - efficient_frontier.png")
    print("        - portfolio_allocation.png")
    print("        - correlation_heatmap.png")
    print("        - cumulative_returns.png")
    print("        - rolling_volatility.png")

    print("\n" + "=" * 78)
    print("DONE. See outputs/ for charts and CSV reports, and README.md for")
    print("the full methodology writeup.")
    print("=" * 78)


if __name__ == "__main__":
    main()
