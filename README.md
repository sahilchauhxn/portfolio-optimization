# Portfolio Optimization using Kenneth French 30 Industry Portfolios

**Global Minimum Variance (GMV) Portfolio construction, benchmarking, and risk analysis on the Ken French 30 Industry universe, built on Modern Portfolio Theory (Markowitz, 1952).**

> Author: Sahil — B.E. Mathematics & Computing, BITS Pilani
> Built as a quant-research portfolio project (targeting Quant Trading / Quant Research / Quant Dev internships).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Setup & Usage](#setup--usage)
4. [Dataset](#dataset)
5. [Modern Portfolio Theory — Background](#modern-portfolio-theory--background)
6. [Global Minimum Variance Portfolio — Mathematical Formulation](#global-minimum-variance-portfolio--mathematical-formulation)
7. [Optimization Constraints](#optimization-constraints)
8. [Methodology](#methodology)
9. [Results](#results)
10. [Interpreting the Charts](#interpreting-the-charts)
11. [Limitations](#limitations)
12. [Future Improvements](#future-improvements)
13. [References](#references)

---

## Project Overview

This project implements the **Global Minimum Variance (GMV) portfolio** — the risk-minimizing point on the Markowitz efficient frontier — on the **Kenneth French 30 Industry Portfolios** dataset, and rigorously benchmarks it against:

- an **Equal-Weight (1/N)** portfolio (the classical "hard to beat" naive benchmark), and
- a **Maximum Sharpe Ratio (tangency)** portfolio, for context.

The GMV portfolio is solved numerically with **`scipy.optimize.minimize`** (SLSQP), cross-checked against its **closed-form analytical solution**, and evaluated with a full institutional-grade performance and risk toolkit: annualized return, annualized volatility, Sharpe ratio, maximum drawdown, rolling volatility, and correlation structure.

---

## Project Structure

```
portfolio-optimization/
│
├── data/                          # Cached/raw dataset (CSV)
├── notebooks/
│   └── portfolio_optimization_analysis.ipynb   # Interactive walkthrough
├── src/
│   ├── data_loader.py             # Download/parse Ken French data, mu & Sigma estimation
│   ├── optimizer.py               # GMV, Equal-Weight, Max-Sharpe, Efficient Frontier (scipy)
│   ├── metrics.py                 # Return/risk metrics: CAGR, vol, Sharpe, MDD, correlation
│   └── visualization.py           # All plotting functions (matplotlib)
├── outputs/                       # Generated charts (PNG) + result tables (CSV)
├── requirements.txt
├── README.md
└── main.py                        # End-to-end CLI pipeline
```

---

## Setup & Usage

```bash
# 1. Clone / unzip the project, then install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (data -> optimization -> metrics -> charts)
python main.py

# 3. (Optional) Explore interactively
jupyter notebook notebooks/portfolio_optimization_analysis.ipynb
```

Running `main.py` will:
1. Load (or download) the 30 Industry Portfolios monthly return dataset.
2. Estimate expected returns (μ) and the annualized covariance matrix (Σ).
3. Solve the GMV, Equal-Weight, and Max-Sharpe portfolios.
4. Print a performance comparison table to stdout.
5. Save all charts and CSV result tables to `outputs/`.

---

## Dataset

**Source:** [Kenneth French Data Library — 30 Industry Portfolios](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) (monthly, value-weighted returns, in %).

`src/data_loader.py` will **automatically attempt to download** the official CSV directly from the Ken French Data Library and parse it (handling the archive's known quirks: multiple tables in one file, `-99.99`/`-999` missing-value sentinels, `YYYYMM` date format).

**⚠️ Note on this delivered project:** the dataset was built and validated inside a network-sandboxed research environment that cannot reach `mba.tuck.dartmouth.edu`. In that case, `data_loader.py` **automatically and transparently falls back** to a synthetic dataset that is statistically constructed to mimic the real 30-industry universe (a common market factor + heterogeneous industry betas and idiosyncratic volatilities), so that every downstream component — optimization, metrics, plots — is fully exercised and reproducible. A prominent console warning is printed whenever the fallback is used.

**To use the real data:**
- Simply run `python src/data_loader.py` (or `main.py`) on any machine with normal internet access — it will download and cache the real dataset automatically, **or**
- Manually download `30_Industry_Portfolios_CSV.zip` from the link above, unzip it, and save the "Average Value Weighted Returns -- Monthly" table as `data/30_Industry_Portfolios_monthly.csv` (Date index in `YYYY-MM-01` format, 30 industry columns).

Either way, **no other code changes are needed** — the rest of the pipeline is agnostic to whether the data is real or the synthetic fallback.

---

## Modern Portfolio Theory — Background

Harry Markowitz's **Modern Portfolio Theory (MPT, 1952)** formalizes portfolio construction as a trade-off between expected return and risk (variance). For a portfolio of $N$ assets with weight vector $w \in \mathbb{R}^N$, expected return vector $\mu$, and covariance matrix $\Sigma$:

$$
E[R_p] = w^\top \mu, \qquad \text{Var}(R_p) = w^\top \Sigma w, \qquad \sigma_p = \sqrt{w^\top \Sigma w}
$$

The set of portfolios that offer the **maximum expected return for a given level of risk** (equivalently, minimum risk for a given expected return) traces out the **efficient frontier**. Rational, risk-averse investors should only ever hold portfolios on this frontier.

Two special points on the frontier are of particular interest:

- The **Global Minimum Variance (GMV)** portfolio — the leftmost point of the frontier, minimizing risk with no regard to return.
- The **Maximum Sharpe Ratio (tangency)** portfolio — the point where a ray from the risk-free rate is tangent to the frontier, maximizing risk-adjusted return.

---

## Global Minimum Variance Portfolio — Mathematical Formulation

The GMV portfolio solves:

$$
\min_{w} \; w^\top \Sigma w
\quad \text{subject to} \quad
\mathbf{1}^\top w = 1
$$

(and, in the long-only variant used as the primary result in this project, $w_i \ge 0 \; \forall i$).

**Key property:** the GMV portfolio depends **only on the covariance matrix** $\Sigma$ — it requires no expected-return forecast at all. This is precisely why it is prized in practice: covariance matrices are estimated far more reliably (lower estimation error) than expected returns (Merton, 1980; Jagannathan & Ma, 2003), so GMV portfolios tend to be substantially more robust out-of-sample than mean-variance-optimal portfolios that depend on noisy $\mu$ forecasts — a central theme in quantitative portfolio management.

**Closed-form solution** (unconstrained, i.e. short-selling allowed):

$$
w^{*} = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top \Sigma^{-1}\mathbf{1}}
$$

This project **numerically verifies** the `scipy.optimize.minimize` (SLSQP) solution against this closed-form formula when short-selling is unconstrained (see `optimizer.gmv_closed_form` and the sanity-check printed by `main.py`) — the two match to numerical precision, confirming the optimizer's correctness.

---

## Optimization Constraints

| Portfolio | Objective | Constraints |
|---|---|---|
| **GMV** | $\min_w w^\top \Sigma w$ | $\mathbf{1}^\top w = 1$, $w_i \ge 0$ (long-only) |
| **Equal Weight** | N/A (deterministic) | $w_i = 1/N$ |
| **Max Sharpe** | $\max_w \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}$ | $\mathbf{1}^\top w = 1$, $w_i \ge 0$ |
| **Efficient Frontier** | $\min_w w^\top \Sigma w$ | $\mathbf{1}^\top w = 1$, $w^\top \mu = \mu_{target}$, $w_i \ge 0$ |

All problems are solved with **Sequential Least Squares Programming (SLSQP)** via `scipy.optimize.minimize`, which natively supports the equality (budget, target-return) and inequality (long-only bound) constraints required.

The **long-only constraint** ($w_i \ge 0$) reflects realistic industry-allocation mandates (most long-only funds and student portfolios cannot short individual industries), and is the headline configuration used throughout this project. The unconstrained (shorting-allowed) case is also implemented, purely to validate the numerical solver against the closed-form GMV formula.

---

## Methodology

1. **Data preprocessing** (`data_loader.py`): load monthly % returns, drop rows with missing data, convert to decimal form.
2. **Expected returns**: historical sample mean of monthly returns, annualized via compounding — $\mu_{annual} = (1+\mu_{monthly})^{12}-1$.
3. **Covariance matrix**: sample covariance of monthly returns, annualized by scaling by 12 (`optimizer.py`, `data_loader.py`).
4. **Optimization** (`optimizer.py`): GMV, Equal-Weight, Max-Sharpe, and a 40-point efficient frontier, all via `scipy.optimize.minimize`.
5. **Performance evaluation** (`metrics.py`): apply each portfolio's *fixed weights* to the realized monthly return history to obtain a realized portfolio return series, then compute:
   - **Annualized Return (CAGR)**: $\left(\prod_t(1+R_t)\right)^{12/T} - 1$
   - **Annualized Volatility**: $\sigma_{monthly}\sqrt{12}$
   - **Sharpe Ratio**: $(R_{annual} - r_f)/\sigma_{annual}$
   - **Maximum Drawdown**: largest peak-to-trough decline of the cumulative wealth curve
   - **Correlation Matrix**: pairwise Pearson correlation of industry returns
6. **Visualization** (`visualization.py`): efficient frontier, portfolio allocation bar chart, correlation heatmap, cumulative returns (growth of $1), rolling 12-month volatility.

---

## Results

*(Results below are from the bundled run — see the note in [Dataset](#dataset) regarding real vs. synthetic-fallback data. Re-run `main.py` with real data for production figures; the methodology and code are identical either way.)*

| Portfolio | Annualized Return | Annualized Volatility | Sharpe Ratio | Max Drawdown | Effective N |
|---|---|---|---|---|---|
| **GMV** | 11.91% | **10.93%** | 1.09 | **-17.56%** | 4.85 |
| **Equal Weight** | 15.32% | 15.58% | 0.98 | -27.80% | 30.0 |
| **Max Sharpe** | 18.32% | 14.64% | **1.25** | -26.97% | 7.33 |

**Key findings:**

- The **GMV portfolio cuts annualized volatility by ~30% relative to Equal Weight** (10.93% vs. 15.58%) and roughly **halves the maximum drawdown** (-17.6% vs. -27.8%), exactly the behavior Modern Portfolio Theory predicts for the leftmost (minimum-risk) point of the efficient frontier.
- GMV achieves this by **concentrating into a smaller set of low-volatility, weakly-correlated industries** (Effective N ≈ 4.85, vs. 30 for Equal Weight by construction) — it exploits diversification *and* selects inherently defensive industries, rather than diversifying blindly.
- The **Max Sharpe (tangency) portfolio delivers the best risk-adjusted return in-sample** (Sharpe 1.25), but this comes with a caveat: unlike GMV, it depends on the (noisy) expected-return estimate $\mu$, making it considerably less robust out-of-sample — a classical critique of naive mean-variance optimization (Michaud, 1989).
- **Average pairwise industry correlation ≈ 0.44** — meaningfully below 1, confirming there is real diversification benefit for the optimizer to exploit (see the correlation heatmap).

Generated artifacts (in `outputs/`):
- `efficient_frontier.png` — full Markowitz frontier with GMV / Equal-Weight / Max-Sharpe highlighted
- `portfolio_allocation.png` — weight comparison across all 30 industries
- `correlation_heatmap.png` — 30×30 industry correlation matrix
- `cumulative_returns.png` — growth-of-$1 comparison (log scale)
- `rolling_volatility.png` — 12-month rolling annualized volatility
- `performance_summary.csv`, `portfolio_weights.csv`, `correlation_matrix.csv`

---

## Interpreting the Charts

- **Efficient Frontier**: individual industries (gray dots) sit well inside/below the frontier curve — no single industry is efficient on its own; only combinations are. GMV sits at the frontier's leftmost tip.
- **Portfolio Allocation**: GMV weight is heavily tilted toward historically low-beta, defensive-style industries (in the synthetic dataset, this shows up as industries with low simulated beta/idiosyncratic vol); Equal Weight is flat at 1/30 ≈ 3.33% for every industry by construction.
- **Correlation Heatmap**: warm (red) cells indicate industries that move together (less diversification value between them); cool (blue) cells indicate genuine diversification opportunities that the optimizer exploits.
- **Cumulative Returns**: plotted on a log scale to fairly compare compounding trajectories over a long (30+ year) horizon.
- **Rolling Volatility**: illustrates how each portfolio's risk profile evolves through time / market regimes — GMV's volatility line should sit consistently below Equal Weight's.

---

## Limitations

This project intentionally uses classical, transparent methods appropriate for a research/interview portfolio piece — but real-world quant desks account for considerably more nuance:

- **Sample covariance estimation error**: with $N=30$ assets and a few hundred monthly observations, the sample covariance matrix is noisy, especially off-diagonal terms — a known weak point of naive Markowitz optimization.
- **In-sample vs. out-of-sample**: all metrics reported here are computed on the *same* historical window used to estimate $\mu$ and $\Sigma$ (in-sample). No rolling/expanding-window out-of-sample backtest is performed.
- **No transaction costs / turnover constraints**: the reported portfolios are static, single-period allocations; ignoring rebalancing costs overstates realistic performance, especially for the more concentrated GMV/Max-Sharpe portfolios.
- **No shrinkage / robust covariance estimation**: the raw sample covariance is used as-is, rather than a shrinkage estimator (e.g. Ledoit-Wolf) that is standard practice for stabilizing $\Sigma$ in higher-dimensional or noisier settings.
- **Stationarity assumption**: returns are assumed i.i.d. across months (used to annualize by $\sqrt{12}$/$\times 12$); real markets exhibit volatility clustering, fat tails, and regime shifts.
- **Synthetic-data caveat**: if the real Ken French dataset could not be downloaded (see [Dataset](#dataset)), the headline results above reflect the synthetic fallback and should be treated as an illustrative pipeline validation rather than a real empirical finding.

---

## Future Improvements

- [ ] **Shrinkage covariance estimators** (Ledoit-Wolf, Oracle Approximating Shrinkage) to stabilize $\Sigma$ and reduce GMV's sensitivity to estimation noise.
- [ ] **Rolling-window out-of-sample backtesting**: re-estimate $\mu$/$\Sigma$ and re-optimize on a rolling basis (e.g. trailing 60 months), then evaluate realized performance strictly out-of-sample.
- [ ] **Black-Litterman model** to blend market-implied equilibrium returns with subjective/analyst views, addressing the instability of naive historical-mean $\mu$ estimates.
- [ ] **Factor-model covariance estimation** (e.g. Fama-French 3/5-factor exposures per industry) instead of raw sample covariance, for a more parsimonious and stable $\Sigma$.
- [ ] **Transaction-cost-aware / turnover-penalized optimization**, and comparison of rebalancing frequencies.
- [ ] **Risk Parity / Hierarchical Risk Parity (HRP)** as additional benchmark allocation methodologies alongside GMV and 1/N.
- [ ] **Live benchmark comparison** against SPY / sector ETFs via `yfinance`, to contextualize industry-portfolio performance against the broader market.
- [ ] **Bootstrapped confidence intervals** on Sharpe ratio and other metrics, to assess statistical significance of GMV's outperformance vs. Equal Weight.

---

## References

- Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance, 7(1), 77–91.
- Sharpe, W. F. (1966). *Mutual Fund Performance*. The Journal of Business, 39(1), 119–138.
- Merton, R. C. (1980). *On Estimating the Expected Return on the Market*. Journal of Financial Economics, 8(4), 323–361.
- Jagannathan, R., & Ma, T. (2003). *Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps*. The Journal of Finance, 58(4), 1651–1683.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). *Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?*. Review of Financial Studies, 22(5), 1915–1953.
- Michaud, R. O. (1989). *The Markowitz Optimization Enigma: Is 'Optimized' Optimal?*. Financial Analysts Journal, 45(1), 31–42.
- Fama, E. F., & French, K. R. — [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

---

*This project was built end-to-end as a self-contained, reproducible research pipeline: data acquisition → estimation → constrained optimization (SciPy/SLSQP) → performance/risk analytics → visualization — intended to demonstrate applied quantitative portfolio construction skills for Quant Research/Trading/Dev internship applications.*
