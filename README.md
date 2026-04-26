# Volatility Forecasting with High-Frequency Intraday Data 

Comparison of GARCH(1,1), HAR-RV, and HAR-RK models for forecasting next-day realized variance using 5-minute intraday equity data.

---

## Research Question

How accurately do GARCH(1,1) and realized-volatility-based HAR models predict next-day realized variance using high-frequency intraday data, and does performance vary across assets?

---

## Data

* Source: Yahoo Finance (via yfinance)
* Frequency: 5-minute intraday bars
* Period: Jan 27 – Apr 23, 2026 (~60 trading days)
* Assets: AAPL, MSFT, SPY, NVDA, GOOGL, GLD
* Train/Test split: 33 days train / 5 days test

---

## Models

* **GARCH(1,1)** — captures volatility clustering from returns
* **HAR-RV** — heterogeneous autoregressive model using realized variance (Corsi, 2009)
* **HAR-RK** — HAR model with Parzen realized kernel estimator, robust to microstructure noise

---

## Forecast Evaluation

Models are compared using:

* **MSE** — mean squared forecast error
* **QLIKE** — volatility-forecast loss robust to measurement error

Lower values indicate better performance.

---

## Results (April 19–23, 2026)

| Asset | GARCH | HAR-RV | HAR-RK | Best   |
|-------|-------|--------|--------|--------|
| AAPL  |      |  ✓      |        | HAR-RV  |
| MSFT  |  ✓     |        |       | GARCH |
| SPY   |   ✓    |        |       | GARCH |
| NVDA  |  ✓     |       |        | GARCH |
| GOOGL |      |        |     ✓   | HAR-RK  |
| GLD   |       |       |    ✓    | HAR-RK |

**Win count (MSE and QLIKE consistent):**
* GARCH: 3 | HAR-RV: 1 | HAR-RK: 2

---

## Key Findings

* No single model dominates across all assets
* GARCH performs best for MSFT, SPY, and NVDA, suggesting that during the test window these assets exhibited fast-moving volatility dynamics where recent shocks dominate persistence.
* HAR-RV outperforms for AAPL, indicating that its volatility process retains multi-horizon structure that benefits from daily/weekly aggregation.
* HAR-RK performs best for GOOGL and GLD, implying that:

** Microstructure noise matters for these assets

** Noise-robust realized measures improve forecast accuracy

* Overall pattern:
** GARCH → short-memory, shock-driven volatility

** HAR-type models → structured, persistent volatility

** HAR-RK → added value when high-frequency noise is non-negligible

---

## Limitations

* The test window (Apr 19–23, 2026) coincided with elevated market volatility driven by macroeconomic policy uncertainty, which may favor GARCH's shock-response mechanism
* Yahoo Finance limits intraday data to the last 60 days — longer evaluation windows and tick-level estimation would require institutional data (WRDS, LOBSTER, TAQ)
* A five-day test window is limited; results are indicative rather than conclusive
* All assets are equities subject to correlated macro shocks — including FX or commodity futures would test performance across more distinct volatility processes

---

## Project Structure

```
volatility-forecasting/
│
├── volatility_multi_asset.py
├── README.md
└── results/
    ├── figures/
    │   └── multi_asset_comparison.png
    └── tables/
        └── multi_asset_results.csv
```

---

## Requirements

```
yfinance
arch
statsmodels
pandas
numpy
matplotlib
```

Install:

```
pip install yfinance arch statsmodels pandas numpy matplotlib
```

---

## Usage

```
python volatility_multi_asset.py
```

---

## References

* Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
* Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2008). Designing Realized Kernels to Measure the Ex Post Variation of Equity Prices in the Presence of Noise. *Econometrica*, 76(6), 1481–1536.
* Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
* Patton, A. J. (2011). Volatility Forecast Comparison Using Imperfect Volatility Proxies. *Journal of Econometrics*, 160(1), 246–256.

---

## Author

Maryam B.
Independent Research Project — Started 2025 — finalized 2026 📊

