"""
Beta estimation following Frazzini & Pedersen (2014) methodology.

Key methodology:
1. Estimate volatilities with 1-year rolling window
2. Estimate correlations with 5-year rolling window
3. Beta = rho * (sigma_i / sigma_m)
4. Shrink toward cross-sectional mean (beta=1) with weight 0.6/0.4
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def estimate_rolling_beta(
    returns: pd.Series,
    market_returns: pd.Series,
    vol_window: int = 12,      # 1 year for monthly data
    corr_window: int = 60,     # 5 years for monthly data
    min_vol_obs: int = 12,     # minimum observations for volatility
    min_corr_obs: int = 36,    # minimum observations for correlation
    shrinkage_weight: float = 0.6,
    shrinkage_target: float = 1.0,
) -> pd.Series:
    """
    Estimate beta for a single asset following FP2014.

    β_hat = ρ * (σ_i / σ_m)  →  then shrink toward 1

    Parameters
    ----------
    returns : pd.Series
        Excess returns of the asset (monthly).
    market_returns : pd.Series
        Excess returns of the market (monthly).
    vol_window : int
        Rolling window for volatility estimation (months).
    corr_window : int
        Rolling window for correlation estimation (months).
    shrinkage_weight : float
        Weight on time-series beta (0.6 in the paper).
    shrinkage_target : float
        Cross-sectional target for shrinkage (1.0 in the paper).

    Returns
    -------
    pd.Series
        Estimated (shrunk) betas.
    """
    # Align indexes
    common_idx = returns.index.intersection(market_returns.index)
    r_i = returns.reindex(common_idx)
    r_m = market_returns.reindex(common_idx)

    # Rolling volatilities (1-year window)
    vol_i = r_i.rolling(window=vol_window, min_periods=min_vol_obs).std()
    vol_m = r_m.rolling(window=vol_window, min_periods=min_vol_obs).std()

    # Rolling correlation (5-year window)
    corr = r_i.rolling(window=corr_window, min_periods=min_corr_obs).corr(r_m)

    # Time-series beta = rho * (sigma_i / sigma_m)
    beta_ts = corr * (vol_i / vol_m)

    # Shrink toward cross-sectional mean
    beta_shrunk = shrinkage_weight * beta_ts + (1 - shrinkage_weight) * shrinkage_target

    return beta_shrunk


def estimate_betas_panel(
    returns_df: pd.DataFrame,
    market_returns: pd.Series,
    vol_window: int = 12,
    corr_window: int = 60,
    shrinkage_weight: float = 0.6,
    shrinkage_target: float = 1.0,
) -> pd.DataFrame:
    """
    Estimate betas for a panel of assets (columns = assets, rows = dates).
    Returns DataFrame of estimated betas with same shape.
    """
    betas = pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float)

    for col in returns_df.columns:
        betas[col] = estimate_rolling_beta(
            returns_df[col], market_returns,
            vol_window=vol_window,
            corr_window=corr_window,
            shrinkage_weight=shrinkage_weight,
            shrinkage_target=shrinkage_target,
        )

    return betas


def estimate_beta_from_portfolios(
    portfolio_returns: pd.DataFrame,
    market_returns: pd.Series,
    vol_window: int = 12,
    corr_window: int = 60,
    shrinkage_weight: float = 0.6,
) -> pd.DataFrame:
    """
    Estimate betas for pre-formed portfolios (e.g., 10 decile portfolios).
    
    Since portfolios are already sorted by beta, the shrinkage target
    is the cross-sectional mean of the portfolio betas at each date.
    
    Returns DataFrame of estimated betas.
    """
    # For portfolios, we use the cross-sectional mean as shrinkage target
    betas_ts = pd.DataFrame(index=portfolio_returns.index,
                            columns=portfolio_returns.columns, dtype=float)

    for col in portfolio_returns.columns:
        r_i = portfolio_returns[col]
        common_idx = r_i.index.intersection(market_returns.index)
        r_i = r_i.reindex(common_idx)
        r_m = market_returns.reindex(common_idx)

        vol_i = r_i.rolling(window=vol_window, min_periods=max(6, vol_window//2)).std()
        vol_m = r_m.rolling(window=vol_window, min_periods=max(6, vol_window//2)).std()
        corr = r_i.rolling(window=corr_window, min_periods=max(24, corr_window//2)).corr(r_m)

        betas_ts[col] = corr * (vol_i / vol_m)

    # Cross-sectional mean at each date
    beta_xs = betas_ts.mean(axis=1)

    # Shrink
    betas_shrunk = shrinkage_weight * betas_ts.add(0) + (1 - shrinkage_weight) * beta_xs.values[:, None]

    return betas_shrunk


def compute_realized_beta(
    returns: pd.Series,
    market_returns: pd.Series,
    window: Optional[int] = None,
) -> float:
    """
    Compute realized (full-sample or rolling) OLS beta.
    """
    common_idx = returns.index.intersection(market_returns.index)
    r_i = returns.reindex(common_idx).dropna()
    r_m = market_returns.reindex(common_idx).reindex(r_i.index).dropna()
    common = r_i.index.intersection(r_m.index)
    r_i, r_m = r_i.reindex(common), r_m.reindex(common)

    if len(r_i) < 10:
        return np.nan

    cov = np.cov(r_i, r_m)
    return cov[0, 1] / cov[1, 1]
