"""
BAB Factor Construction following Frazzini & Pedersen (2014).

The BAB factor:
1. Ranks securities by estimated beta
2. Forms long (low-beta) and short (high-beta) portfolios with rank-based weights
3. Leverages the long side to beta=1, de-leverages the short side to beta=1
4. BAB = (1/β_L)(r_L - rf) - (1/β_H)(r_H - rf)

The result is a self-financing, market-neutral portfolio.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def construct_bab_from_portfolios(
    portfolio_returns: pd.DataFrame,
    portfolio_betas: pd.DataFrame,
    rf_rate: pd.Series,
    n_low: Optional[int] = None,
    n_high: Optional[int] = None,
) -> dict:
    """
    Construct BAB factor from pre-sorted beta portfolios.

    Uses the median split: portfolios below median beta go into the low-beta
    portfolio, those above go into the high-beta portfolio.

    Parameters
    ----------
    portfolio_returns : DataFrame
        Monthly returns of beta-sorted portfolios (columns P1 to P10).
    portfolio_betas : DataFrame
        Estimated betas for each portfolio (same structure).
    rf_rate : Series
        Monthly risk-free rate.

    Returns
    -------
    dict with keys:
        'bab_returns': pd.Series of BAB monthly returns
        'low_beta_returns': returns of the low-beta portfolio
        'high_beta_returns': returns of the high-beta portfolio
        'beta_low': ex-ante beta of low portfolio
        'beta_high': ex-ante beta of high portfolio
    """
    # Align all data
    common_idx = (portfolio_returns.index
                  .intersection(portfolio_betas.index)
                  .intersection(rf_rate.index))
    ret = portfolio_returns.reindex(common_idx)
    betas = portfolio_betas.reindex(common_idx)
    rf = rf_rate.reindex(common_idx)

    n_ports = ret.shape[1]
    if n_low is None:
        n_low = n_ports // 2
    if n_high is None:
        n_high = n_ports // 2

    # Low-beta: first n_low portfolios, High-beta: last n_high portfolios
    low_cols = ret.columns[:n_low]
    high_cols = ret.columns[-n_high:]

    # Rank-based weights within each group
    # For low-beta: weight inversely proportional to beta rank (lower beta → higher weight)
    # For high-beta: weight proportional to beta rank (higher beta → higher weight)

    # Simple approach for pre-formed deciles: equal-weight within each half
    # (since they're already sorted, rank-weighting within deciles is less critical)
    r_L = ret[low_cols].mean(axis=1)
    r_H = ret[high_cols].mean(axis=1)

    beta_L = betas[low_cols].mean(axis=1)
    beta_H = betas[high_cols].mean(axis=1)

    # Clip betas to avoid extreme leverage
    beta_L = beta_L.clip(lower=0.1)
    beta_H = beta_H.clip(lower=0.1)

    # BAB = (1/β_L)(r_L - rf) - (1/β_H)(r_H - rf)
    excess_L = r_L - rf
    excess_H = r_H - rf

    bab_returns = (1.0 / beta_L) * excess_L - (1.0 / beta_H) * excess_H

    return {
        'bab_returns': bab_returns,
        'low_beta_returns': r_L,
        'high_beta_returns': r_H,
        'low_beta_excess': excess_L,
        'high_beta_excess': excess_H,
        'beta_low': beta_L,
        'beta_high': beta_H,
        'rf': rf,
    }


def construct_bab_rank_weighted(
    portfolio_returns: pd.DataFrame,
    portfolio_betas: pd.DataFrame,
    rf_rate: pd.Series,
) -> dict:
    """
    Construct BAB factor using rank-weighted portfolios, closer to the paper.

    For N portfolios sorted by beta:
    - Compute rank z_i = rank(beta_i)
    - Low-beta weights: proportional to (z_bar - z_i) for z_i < z_bar
    - High-beta weights: proportional to (z_i - z_bar) for z_i > z_bar
    - Both normalized to sum to 1
    """
    common_idx = (portfolio_returns.index
                  .intersection(portfolio_betas.index)
                  .intersection(rf_rate.index))
    ret = portfolio_returns.reindex(common_idx)
    betas = portfolio_betas.reindex(common_idx)
    rf = rf_rate.reindex(common_idx)

    n = ret.shape[1]

    bab_list = []
    beta_L_list = []
    beta_H_list = []
    rL_list = []
    rH_list = []

    for date in common_idx:
        b = betas.loc[date].values
        r = ret.loc[date].values
        rf_t = rf.loc[date]

        if np.any(np.isnan(b)) or np.any(np.isnan(r)) or np.isnan(rf_t):
            bab_list.append(np.nan)
            beta_L_list.append(np.nan)
            beta_H_list.append(np.nan)
            rL_list.append(np.nan)
            rH_list.append(np.nan)
            continue

        # Ranks (1 to n)
        ranks = np.argsort(np.argsort(b)) + 1  # rank based on betas
        z = ranks.astype(float)
        z_bar = z.mean()

        # Weights
        w_low = np.maximum(z_bar - z, 0)
        w_high = np.maximum(z - z_bar, 0)

        # Normalize
        if w_low.sum() > 0:
            w_low = w_low / w_low.sum()
        if w_high.sum() > 0:
            w_high = w_high / w_high.sum()

        # Portfolio returns and betas
        r_L = np.dot(w_low, r)
        r_H = np.dot(w_high, r)
        b_L = np.dot(w_low, b)
        b_H = np.dot(w_high, b)

        # Clip
        b_L = max(b_L, 0.1)
        b_H = max(b_H, 0.1)

        # BAB return
        bab_t = (1.0 / b_L) * (r_L - rf_t) - (1.0 / b_H) * (r_H - rf_t)

        bab_list.append(bab_t)
        beta_L_list.append(b_L)
        beta_H_list.append(b_H)
        rL_list.append(r_L)
        rH_list.append(r_H)

    return {
        'bab_returns': pd.Series(bab_list, index=common_idx, name='BAB'),
        'low_beta_returns': pd.Series(rL_list, index=common_idx),
        'high_beta_returns': pd.Series(rH_list, index=common_idx),
        'beta_low': pd.Series(beta_L_list, index=common_idx),
        'beta_high': pd.Series(beta_H_list, index=common_idx),
        'rf': rf,
    }


def compute_portfolio_stats(
    returns: pd.Series,
    rf: pd.Series,
    annualize: bool = True,
) -> dict:
    """Compute key portfolio statistics."""
    excess = returns - rf.reindex(returns.index)
    excess = excess.dropna()

    n = len(excess)
    mean_ret = excess.mean()
    vol = excess.std()
    sr = mean_ret / vol if vol > 0 else np.nan

    if annualize:
        mean_ret_ann = mean_ret * 12
        vol_ann = vol * np.sqrt(12)
        sr_ann = sr * np.sqrt(12)
    else:
        mean_ret_ann = mean_ret
        vol_ann = vol
        sr_ann = sr

    return {
        'mean_monthly': mean_ret,
        'mean_annual': mean_ret_ann,
        'vol_monthly': vol,
        'vol_annual': vol_ann,
        'sharpe_monthly': sr,
        'sharpe_annual': sr_ann,
        'n_obs': n,
        'min': excess.min(),
        'max': excess.max(),
        'skew': excess.skew(),
        'kurtosis': excess.kurtosis(),
    }
