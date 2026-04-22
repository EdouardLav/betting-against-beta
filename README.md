# Betting Against Beta — Replication of Frazzini & Pedersen (2014)

Replication of the BAB factor from Frazzini & Pedersen (2014, JFE) using AQR's official factor data (1931–2025). The paper shows that the Security Market Line is too flat: low-beta stocks earn higher risk-adjusted returns than high-beta stocks, due to leverage constraints pushing investors toward risky assets.

I replicate the main US results (performance stats, factor regressions, sub-period robustness), extend the analysis to 24 international equity markets, and build a global diversified BAB portfolio as an extension.

## Results vs. Paper

| | Paper (1926–2012) | This replication (1931–2025) |
|--|--|--|
| BAB Sharpe | 0.78 | 0.70 |
| BAB t-stat | 7.12 | 6.86 |
| CAPM α | 0.73%/mo | 0.71%/mo |
| FF3 α | 0.73%/mo | 0.70%/mo |
| Carhart α | 0.55%/mo | 0.50%/mo |
| Countries with positive SR | 18/19 | 24/24 |

Small differences come from the extended sample (13 extra years including COVID, low-vol drawdown of 2018) and AQR's updated data construction.

## Methodology

Beta estimation follows FP2014 §3.1: decomposed beta β = ρ × (σ_i / σ_m) with 1-year vol window, 5-year correlation window, and Bayesian shrinkage toward 1 (weight 0.6 on time-series estimate, 0.4 on prior). BAB construction per §3.2: rank-weighted long/short portfolios, each side levered to β = 1.

All regressions use Newey-West standard errors (6 lags).

**Carhart loadings:** MKT = −0.034 (t = −0.96), SMB = 0.045 (t = 0.51), HML = 0.100 (t = 1.15), UMD = 0.233 (t = 4.02). The momentum loading is notable — BAB picks up some UMD exposure, which is why alpha drops from 0.71% to 0.50% in Carhart.

## Sub-Period Robustness

| Period | Sharpe | t-stat |
|--|--|--|
| 1931–1950 | 0.31 | 1.37 |
| 1951–1970 | 1.11 | 4.98 |
| 1971–1990 | 1.13 | 5.07 |
| 1991–2012 | 0.71 | 3.25 |
| 2012–2025 (OOS) | 0.74 | 2.74 |

Positive in every sub-period. The out-of-sample Sharpe (0.74) is actually slightly higher than in-sample 1991–2012, which is reassuring for persistence.

## International Analysis

24/24 countries have positive BAB Sharpe ratios. Top: Canada (1.29), France (1.05), Israel (1.00). Weakest: Ireland (0.02), Austria (0.24), Japan (0.26). Cross-country BAB correlations average 0.10–0.30, which motivates the global extension.

## Extension: Global Diversified BAB

Risk-parity (inverse-vol weighted) portfolio across 12 major markets, 1990–2025:

| | Sharpe | Max Drawdown |
|--|--|--|
| US only | 0.64 | −54.6% |
| Global diversified | 1.34 | −23.8% |

Low cross-country correlations do the heavy lifting here — you roughly double the Sharpe and halve the drawdown.

## Data

AQR Capital Management factor data ([link](https://www.aqr.com/Insights/Datasets/Betting-Against-Beta-Equity-Factors-Monthly)). Download the Excel file and place it in `data/`. A full stock-level replication would require CRSP via WRDS.

## Running

```bash
pip install -r requirements.txt
python main.py
```

Generates 8 figures in `figures/` and regression results in `results/`.

## Structure

```
main.py                 # Full analysis pipeline
src/
  data_loader.py        # AQR data parsing
  beta_estimation.py    # Rolling beta with shrinkage
  bab_construction.py   # BAB portfolio construction
  performance.py        # Regressions & performance stats
figures/                # Output charts (8 PNGs)
results/                # Output CSVs
```

## References

- Frazzini, A. & Pedersen, L.H. (2014). Betting Against Beta. *Journal of Financial Economics*, 111, 1–25.
- Black, F. (1972). Capital Market Equilibrium with Restricted Borrowing. *Journal of Business*, 45(3), 444–455.
- Frazzini, A., Kabiller, D. & Pedersen, L.H. (2018). Buffett's Alpha. *Financial Analysts Journal*, 74(4), 35–55.
