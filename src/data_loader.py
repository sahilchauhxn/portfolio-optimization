"""
data_loader.py
===============
Handles acquisition and preprocessing of the Kenneth French 30 Industry
Portfolios dataset (monthly returns, value-weighted).

Data source (official):
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    File: "30 Industry Portfolios" -> 30_Industry_Portfolios_CSV.zip

The raw CSV published by Ken French's data library has a distinctive,
slightly awkward structure that we need to parse carefully:

    1. A few lines of free-text header/description.
    2. A table: "Average Value Weighted Returns -- Monthly"
       (this is the table we want)
    3. A table: "Average Equal Weighted Returns -- Monthly"
    4. Several more tables (annual returns, number of firms, avg firm size,
       declared each with their own text header).
    5. Missing observations are coded as -99.99 or -999.

Because the data library is hosted on a domain that is frequently blocked
by sandboxed/offline execution environments (as is the case here), this
module:
    (a) Implements a *correct* downloader/parser that will work out of the
        box on a machine with normal internet access (run this on your own
        laptop and it will fetch the real data), AND
    (b) Falls back to a clearly-labeled SYNTHETIC dataset generator that
        statistically mimics the real 30-industry universe (realistic
        volatilities, a common market factor, industry clustering) so the
        rest of the pipeline (optimizer, metrics, plots) can be developed,
        tested and demonstrated end-to-end even without internet access.

If the fallback is used, a prominent warning is printed/logged and the
README explains exactly how to obtain and drop in the real data file.
"""

from __future__ import annotations

import io
import os
import zipfile
import logging
from typing import Optional

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Official Ken French Data Library URL for the 30 Industry Portfolios (CSV, monthly+annual)
KEN_FRENCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/30_Industry_Portfolios_CSV.zip"
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_CSV_PATH = os.path.join(DATA_DIR, "30_Industry_Portfolios_raw.csv")
CLEAN_CSV_PATH = os.path.join(DATA_DIR, "30_Industry_Portfolios_monthly.csv")

MISSING_CODES = (-99.99, -999)


def _download_zip(url: str = KEN_FRENCH_URL, timeout: int = 30) -> Optional[bytes]:
    """
    Attempt to download the raw zip archive from the Ken French Data Library.

    Returns the raw bytes of the CSV file inside the archive, or None if the
    download fails for any reason (no internet, blocked domain, timeout,
    server error, etc). We deliberately swallow the exception here and let
    the caller decide how to handle the fallback.
    """
    try:
        logger.info("Attempting to download 30 Industry Portfolios from Ken French Data Library...")
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # The CSV file inside the archive is typically named
            # "30_Industry_Portfolios.CSV" (case can vary).
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise FileNotFoundError("No CSV file found inside downloaded zip archive.")
            with zf.open(csv_names[0]) as f:
                raw_bytes = f.read()
        logger.info("Download successful.")
        return raw_bytes
    except Exception as exc:  # noqa: BLE001 - we want a broad, safe fallback
        logger.warning(f"Download failed ({exc.__class__.__name__}: {exc}). Will use fallback data.")
        return None


def _parse_ken_french_csv(raw_bytes: bytes) -> pd.DataFrame:
    """
    Parse the raw Ken French CSV bytes and extract ONLY the first table,
    'Average Value Weighted Returns -- Monthly', which is the standard
    convention used for portfolio construction (value-weighting is what
    the classic Fama-French / industry portfolio literature uses).

    The parser is defensive: it scans line-by-line for the start of the
    monthly table (a line whose first token is a 6-digit YYYYMM date) and
    stops at the first blank line or non-numeric row, since the file
    switches to annual data / other tables after that.
    """
    text = raw_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        first_tok = line.split(",")[0].strip()
        if first_tok.isdigit() and len(first_tok) == 6:
            header_idx = i - 1  # the line just above the first data row is the header
            break
    if header_idx is None:
        raise ValueError("Could not locate the start of the monthly returns table in the CSV.")

    header_line = lines[header_idx]
    columns = [c.strip() for c in header_line.split(",")]
    columns[0] = "Date"

    data_rows = []
    for line in lines[header_idx + 1:]:
        first_tok = line.split(",")[0].strip()
        if first_tok.isdigit() and len(first_tok) == 6:
            data_rows.append([v.strip() for v in line.split(",")])
        else:
            # First non-matching line marks the end of the monthly table
            break

    df = pd.DataFrame(data_rows, columns=columns)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df = df.set_index("Date")
    df = df.apply(pd.to_numeric, errors="coerce")

    # Replace Ken French's missing-data sentinel codes with NaN
    df = df.replace(list(MISSING_CODES), np.nan)

    return df


def _generate_synthetic_industry_data(
    n_months: int = 420,
    n_industries: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic monthly-return panel that mimics the statistical
    properties of the Ken French 30 Industry Portfolios:
        - A single dominant market factor (captures the fact that industry
          returns are highly correlated through the business cycle).
        - Industry-specific idiosyncratic volatility on top of the factor.
        - Heterogeneous industry betas and volatilities (e.g. "Utilities"-
          type industries have low beta/vol, "Autos"/"Fin"-type have higher
          beta/vol), so the resulting covariance matrix is non-trivial and
          produces a meaningful (non-degenerate) GMV solution.

    THIS IS A FALLBACK USED ONLY WHEN THE REAL DATA CANNOT BE DOWNLOADED
    (e.g. due to a sandboxed/offline execution environment). It is clearly
    labeled as such everywhere it is used. Replace with real data by
    running this module with normal internet access, or by manually placing
    the official CSV in data/30_Industry_Portfolios_monthly.csv
    """
    rng = np.random.default_rng(seed)

    industry_names = [
        "Food", "Beer", "Smoke", "Games", "Books", "Hshld", "Clths", "Hlth",
        "Chems", "Txtls", "Cnstr", "Steel", "FabPr", "ElcEq", "Autos",
        "Carry", "Mines", "Coal", "Oil", "Util", "Telcm", "Servs", "BusEq",
        "Paper", "Trans", "Whlsl", "Rtail", "Meals", "Fin", "Other",
    ][:n_industries]

    # Heterogeneous market betas (defensive industries lower, cyclical higher)
    betas = rng.normal(loc=1.0, scale=0.35, size=n_industries).clip(0.3, 2.0)
    # Heterogeneous idiosyncratic monthly vol (in decimal, e.g. 0.04 = 4%/mo)
    idio_vol = rng.uniform(0.025, 0.075, size=n_industries)
    # Heterogeneous small monthly alpha (drift), annualized ~4%-12%
    monthly_alpha = rng.uniform(0.003, 0.010, size=n_industries)

    # Market factor: monthly mean ~0.8%, monthly vol ~4.3% (roughly matches
    # long-run equity market statistics)
    market = rng.normal(loc=0.008, scale=0.043, size=n_months)

    idio = rng.normal(loc=0.0, scale=1.0, size=(n_months, n_industries)) * idio_vol
    returns = (monthly_alpha + np.outer(market, betas)) * 100 + idio * 100  # convert to % like real data

    today_month_start = pd.Timestamp.today().normalize().replace(day=1)
    last_full_month = today_month_start - pd.offsets.MonthBegin(1)
    dates = pd.date_range(end=last_full_month, periods=n_months, freq="MS")
    df = pd.DataFrame(returns, index=dates, columns=industry_names)
    return df.round(2)


def load_industry_returns(
    force_refresh: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main entry point. Loads (downloading if necessary) the Ken French 30
    Industry Portfolios monthly value-weighted return panel, in PERCENT
    (i.e. 1.23 means +1.23% that month), indexed by month-start Timestamp.

    Parameters
    ----------
    force_refresh : bool
        If True, ignore any cached local CSV and re-download / re-generate.
    start_date, end_date : str, optional
        Optional 'YYYY-MM' bounds to slice the returned panel.

    Returns
    -------
    pd.DataFrame
        Rows = months, Columns = 30 industries, values = monthly % returns.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if not force_refresh and os.path.exists(CLEAN_CSV_PATH):
        logger.info(f"Loading cached dataset from {CLEAN_CSV_PATH}")
        df = pd.read_csv(CLEAN_CSV_PATH, index_col=0, parse_dates=True)
    else:
        raw_bytes = _download_zip()
        if raw_bytes is not None:
            try:
                df = _parse_ken_french_csv(raw_bytes)
                logger.info(f"Parsed real Ken French dataset: {df.shape[0]} months x {df.shape[1]} industries.")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Parsing failed ({exc}); falling back to synthetic data.")
                df = _generate_synthetic_industry_data()
                _print_fallback_warning()
        else:
            df = _generate_synthetic_industry_data()
            _print_fallback_warning()

        df.to_csv(CLEAN_CSV_PATH)
        logger.info(f"Saved cleaned dataset to {CLEAN_CSV_PATH}")

    # Drop any months with missing data across all industries (rare, but Ken
    # French's series does have a few early NaNs for some industries)
    df = df.dropna(how="any")

    if start_date is not None:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.Timestamp(end_date)]

    return df


def _print_fallback_warning() -> None:
    print("\n" + "=" * 78)
    print("WARNING: Could not reach the Ken French Data Library (this sandboxed")
    print("environment has restricted network egress). Using a SYNTHETIC dataset")
    print("that mimics the statistical structure of the real 30 Industry")
    print("Portfolios (market factor + heterogeneous betas/vols) so the full")
    print("pipeline can still run end-to-end.")
    print()
    print("To use REAL data: run `python src/data_loader.py` on a machine with")
    print("normal internet access, OR manually download:")
    print(f"  {KEN_FRENCH_URL}")
    print("unzip it, and place the CSV's monthly value-weighted-returns table")
    print(f"as {CLEAN_CSV_PATH}")
    print("=" * 78 + "\n")


def compute_expected_returns(returns_df: pd.DataFrame, annualize: bool = True) -> pd.Series:
    """
    Expected return per asset, estimated as the historical sample mean of
    monthly returns (the standard, simplest estimator used in classical
    Markowitz mean-variance optimization).

        mu_i = (1/T) * sum_t r_{i,t}

    If annualize=True, compounds the monthly mean up to an annual figure:
        mu_annual = (1 + mu_monthly)^12 - 1
    """
    mu_monthly = returns_df.mean() / 100.0  # convert from % to decimal
    if annualize:
        return (1 + mu_monthly) ** 12 - 1
    return mu_monthly


def compute_covariance_matrix(returns_df: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """
    Sample covariance matrix of monthly returns.

        Sigma_{ij} = (1/(T-1)) * sum_t (r_{i,t} - mu_i)(r_{j,t} - mu_j)

    If annualize=True, scales by 12 (since returns are ~i.i.d. across
    months under the classical assumption, variance scales linearly with
    time, i.e. Sigma_annual = 12 * Sigma_monthly).
    """
    returns_decimal = returns_df / 100.0
    cov_monthly = returns_decimal.cov()
    if annualize:
        return cov_monthly * 12
    return cov_monthly


if __name__ == "__main__":
    # Running this file directly attempts a fresh download/parse and reports
    # basic diagnostics -- useful as a standalone sanity check.
    data = load_industry_returns(force_refresh=True)
    print(data.tail())
    print(f"\nShape: {data.shape}")
    print(f"Date range: {data.index.min().date()} -> {data.index.max().date()}")
