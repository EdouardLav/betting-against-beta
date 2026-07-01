"""
Performance analysis for BAB factor replication.

Includes:
- Factor regression analysis (CAPM, FF3, FF4/Carhart, FF5)
- Portfolio performance metrics
- Comparison with original paper results
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import statsmodels.api as sm


# ──────────────────────────────────────────────────────────
# Factor Regressions
# ──────────────────────────────────────────────────────────

def run_factor_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    factor_names: Optional[List[str]] = None,
    newey_west_lags: int = 6,
) -> dict:

    if factor_names:
        X = factors[factor_names].copy()
    else:
        X = factors.copy()

    # Align
    common_idx = returns.index.intersection(X.index)
    y = returns.reindex(common_idx).dropna()
    X = X.reindex(y.index).dropna()
    common_idx = y.index.intersection(X.index)
    y = y.reindex(common_idx)
    X = X.reindex(common_idx)

    # Drop any remaining NaN
    mask = ~(y.isna() | X.isna().any(axis=1))
    y = y[mask]
    X = X[mask]

    if len(y) < 10:
        return {'error': 'Insufficient observations'}

    # Add constant (alpha)
    X_const = sm.add_constant(X)

    # OLS with Newey-West
    model = sm.OLS(y, X_const)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags})

    # Extract results
    alpha = results.params['const']
    alpha_tstat = results.tvalues['const']
    alpha_pval = results.pvalues['const']

    betas = {}
    beta_tstats = {}
    for col in X.columns:
        betas[col] = results.params[col]
        beta_tstats[col] = results.tvalues[col]

    return {
        'alpha': alpha,
        'alpha_tstat': alpha_tstat,
        'alpha_pval': alpha_pval,
        'betas': betas,
        'beta_tstats': beta_tstats,
        'r_squared': results.rsquared,
        'adj_r_squared': results.rsquared_adj,
        'n_obs': results.nobs,
        'model': results,
    }


def run_all_regressions(
    bab_returns: pd.Series,
    ff3: pd.DataFrame,
    ff5: Optional[pd.DataFrame] = None,
    mom: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Run CAPM, FF3, Carhart (FF3+Mom), and FF5 regressions on BAB returns.
    Returns a summary DataFrame.
    """
    results = []

    # Get market excess return
    mkt_col = [c for c in ff3.columns if 'Mkt' in c or 'MKT' in c or 'Mkt-RF' in c]
    if mkt_col:
        mkt_col = mkt_col[0]
    else:
        mkt_col = ff3.columns[0]

    # 1) CAPM
    capm = run_factor_regression(bab_returns, ff3, [mkt_col])
    results.append({
        'Model': 'CAPM',
        'Alpha (monthly %)': capm['alpha'] * 100,
        'Alpha t-stat': capm['alpha_tstat'],
        'Mkt Beta': capm['betas'].get(mkt_col, np.nan),
        'Mkt t-stat': capm['beta_tstats'].get(mkt_col, np.nan),
        'R²': capm['r_squared'],
        'N': capm['n_obs'],
    })

    # 2) FF3
    ff3_factors = [mkt_col, 'SMB', 'HML']
    ff3_available = [f for f in ff3_factors if f in ff3.columns]
    reg_ff3 = run_factor_regression(bab_returns, ff3, ff3_available)
    results.append({
        'Model': 'FF3',
        'Alpha (monthly %)': reg_ff3['alpha'] * 100,
        'Alpha t-stat': reg_ff3['alpha_tstat'],
        'Mkt Beta': reg_ff3['betas'].get(mkt_col, np.nan),
        'Mkt t-stat': reg_ff3['beta_tstats'].get(mkt_col, np.nan),
        'R²': reg_ff3['r_squared'],
        'N': reg_ff3['n_obs'],
    })

    # 3) Carhart (FF3 + Momentum)
    if mom is not None:
        mom_col = mom.columns[0] if len(mom.columns) == 1 else [c for c in mom.columns if 'Mom' in c or 'UMD' in c or 'WML' in c]
        if isinstance(mom_col, list):
            mom_col = mom_col[0] if mom_col else mom.columns[0]

        combined = ff3.join(mom[[mom_col]].rename(columns={mom_col: 'UMD'}), how='inner')
        carhart_factors = ff3_available + ['UMD']
        reg_carhart = run_factor_regression(bab_returns, combined, carhart_factors)
        results.append({
            'Model': 'Carhart (FF3+Mom)',
            'Alpha (monthly %)': reg_carhart['alpha'] * 100,
            'Alpha t-stat': reg_carhart['alpha_tstat'],
            'Mkt Beta': reg_carhart['betas'].get(mkt_col, np.nan),
            'Mkt t-stat': reg_carhart['beta_tstats'].get(mkt_col, np.nan),
            'R²': reg_carhart['r_squared'],
            'N': reg_carhart['n_obs'],
        })

    # 4) FF5
    if ff5 is not None:
        mkt5 = [c for c in ff5.columns if 'Mkt' in c or 'MKT' in c][0]
        ff5_factors = [c for c in ff5.columns if c != 'RF']
        reg_ff5 = run_factor_regression(bab_returns, ff5, ff5_factors)
        results.append({
            'Model': 'FF5',
            'Alpha (monthly %)': reg_ff5['alpha'] * 100,
            'Alpha t-stat': reg_ff5['alpha_tstat'],
            'Mkt Beta': reg_ff5['betas'].get(mkt5, np.nan),
            'Mkt t-stat': reg_ff5['beta_tstats'].get(mkt5, np.nan),
            'R²': reg_ff5['r_squared'],
            'N': reg_ff5['n_obs'],
        })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────
# Decile-level analysis (reproducing Table 3 of paper)
# ──────────────────────────────────────────────────────────

def analyze_decile_portfolios(
    portfolio_returns: pd.DataFrame,
    market_excess: pd.Series,
    rf: pd.Series,
    ff_factors: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Reproduce Table 3: statistics for each beta-sorted decile portfolio.
    """
    results = []

    for col in portfolio_returns.columns:
        ret = portfolio_returns[col]
        common_idx = ret.index.intersection(market_excess.index).intersection(rf.index)
        ret_c = ret.reindex(common_idx)
        mkt_c = market_excess.reindex(common_idx)
        rf_c = rf.reindex(common_idx)

        # Excess return
        excess_ret = ret_c  # French data already excess returns for some portfolios

        mean_excess = excess_ret.mean() * 100  # monthly %
        vol = excess_ret.std() * np.sqrt(12) * 100  # annualized %
        sr = (excess_ret.mean() / excess_ret.std()) * np.sqrt(12) if excess_ret.std() > 0 else np.nan

        # CAPM alpha
        mask = ~(excess_ret.isna() | mkt_c.isna())
        y = excess_ret[mask]
        X = sm.add_constant(mkt_c[mask])
        try:
            reg = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
            capm_alpha = reg.params['const'] * 100
            capm_alpha_t = reg.tvalues['const']
            realized_beta = reg.params.iloc[1] if len(reg.params) > 1 else np.nan
        except Exception:
            capm_alpha = np.nan
            capm_alpha_t = np.nan
            realized_beta = np.nan

        # FF3 alpha
        ff3_alpha = np.nan
        ff3_alpha_t = np.nan
        if ff_factors is not None:
            try:
                factor_cols = [c for c in ff_factors.columns if c != 'RF']
                factors_c = ff_factors[factor_cols].reindex(common_idx)
                mask2 = ~(excess_ret.isna() | factors_c.isna().any(axis=1))
                y2 = excess_ret[mask2]
                X2 = sm.add_constant(factors_c[mask2])
                reg2 = sm.OLS(y2, X2).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
                ff3_alpha = reg2.params['const'] * 100
                ff3_alpha_t = reg2.tvalues['const']
            except Exception:
                pass

        results.append({
            'Portfolio': col,
            'Excess Return (%)': mean_excess,
            'CAPM Alpha (%)': capm_alpha,
            'CAPM Alpha t': capm_alpha_t,
            'FF3 Alpha (%)': ff3_alpha,
            'FF3 Alpha t': ff3_alpha_t,
            'Realized Beta': realized_beta,
            'Volatility (ann %)': vol,
            'Sharpe Ratio': sr,
        })

    return pd.DataFrame(results).set_index('Portfolio')


# ──────────────────────────────────────────────────────────
# Comparison with paper
# ──────────────────────────────────────────────────────────

# Table 3 values from the paper (US equities, 1926-2012)
PAPER_TABLE3 = {
    'P1':  {'excess_ret': 0.91, 'capm_alpha': 0.52, 'ff3_alpha': 0.40, 'sr': 0.70, 'beta_ex_ante': 0.64},
    'P2':  {'excess_ret': 0.98, 'capm_alpha': 0.48, 'ff3_alpha': 0.35, 'sr': 0.63, 'beta_ex_ante': 0.79},
    'P3':  {'excess_ret': 1.00, 'capm_alpha': 0.42, 'ff3_alpha': 0.26, 'sr': 0.57, 'beta_ex_ante': 0.88},
    'P4':  {'excess_ret': 1.03, 'capm_alpha': 0.39, 'ff3_alpha': 0.21, 'sr': 0.54, 'beta_ex_ante': 0.97},
    'P5':  {'excess_ret': 1.05, 'capm_alpha': 0.34, 'ff3_alpha': 0.13, 'sr': 0.49, 'beta_ex_ante': 1.05},
    'P6':  {'excess_ret': 1.10, 'capm_alpha': 0.34, 'ff3_alpha': 0.11, 'sr': 0.48, 'beta_ex_ante': 1.12},
    'P7':  {'excess_ret': 1.05, 'capm_alpha': 0.22, 'ff3_alpha': 0.03, 'sr': 0.42, 'beta_ex_ante': 1.21},
    'P8':  {'excess_ret': 1.08, 'capm_alpha': 0.21, 'ff3_alpha': -0.06, 'sr': 0.41, 'beta_ex_ante': 1.31},
    'P9':  {'excess_ret': 1.06, 'capm_alpha': 0.10, 'ff3_alpha': -0.22, 'sr': 0.36, 'beta_ex_ante': 1.44},
    'P10': {'excess_ret': 0.97, 'capm_alpha': -0.10, 'ff3_alpha': -0.49, 'sr': 0.28, 'beta_ex_ante': 1.70},
    'BAB': {'excess_ret': 0.70, 'capm_alpha': 0.73, 'ff3_alpha': 0.73, 'sr': 0.78},
}


def compare_with_paper(my_results: pd.DataFrame) -> pd.DataFrame:
    """Create comparison table between replication and paper."""
    comparison = []

    for port_name, paper_vals in PAPER_TABLE3.items():
        if port_name in my_results.index:
            my = my_results.loc[port_name]
            comparison.append({
                'Portfolio': port_name,
                'Paper Excess Ret (%)': paper_vals.get('excess_ret', np.nan),
                'My Excess Ret (%)': my.get('Excess Return (%)', np.nan),
                'Paper CAPM α (%)': paper_vals.get('capm_alpha', np.nan),
                'My CAPM α (%)': my.get('CAPM Alpha (%)', np.nan),
                'Paper SR': paper_vals.get('sr', np.nan),
                'My SR': my.get('Sharpe Ratio', np.nan),
            })

    return pd.DataFrame(comparison).set_index('Portfolio')


# ──────────────────────────────────────────────────────────
# Sub-period analysis
# ──────────────────────────────────────────────────────────

def subperiod_analysis(
    bab_returns: pd.Series,
    periods: Optional[List[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Analyze BAB performance across sub-periods.
    """
    if periods is None:
        periods = [
            ('1930', '1950'),
            ('1950', '1970'),
            ('1970', '1990'),
            ('1990', '2010'),
            ('2010', '2025'),
        ]

    results = []
    for start, end in periods:
        sub = bab_returns.loc[start:end].dropna()
        if len(sub) < 12:
            continue
        mean_m = sub.mean()
        vol_m = sub.std()
        sr = (mean_m / vol_m * np.sqrt(12)) if vol_m > 0 else np.nan

        # t-stat
        t_stat = mean_m / (vol_m / np.sqrt(len(sub))) if vol_m > 0 else np.nan

        results.append({
            'Period': f'{start}–{end}',
            'Mean (monthly %)': mean_m * 100,
            't-stat': t_stat,
            'Vol (ann %)': vol_m * np.sqrt(12) * 100,
            'Sharpe (ann)': sr,
            'N months': len(sub),
        })

    return pd.DataFrame(results).set_index('Period')
