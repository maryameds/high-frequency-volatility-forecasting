# Volatility Forecasting with High-Frequency Intraday Data
# Multi-Asset Comparison: GARCH(1,1) vs HAR-RV vs HAR-RK
# Author: Maryam B.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from arch import arch_model
import yfinance as yf
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)


# CONFIGURATION


TICKERS = ["AAPL", "MSFT", "SPY", "NVDA", "GOOGL", "GLD"]
TEST_DAYS = 5


# HELPER FUNCTIONS


def get_data(ticker):
    raw = yf.download(
        ticker, period="60d", interval="5m",
        auto_adjust=True, progress=False
    )
    prices = raw["Close"].dropna().squeeze()
    prices.index = prices.index.tz_localize(None)
    return prices


def compute_log_returns(prices):
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def compute_rv(returns):
    rv = (returns ** 2).resample("1D").sum()
    rv = rv.dropna()
    rv = rv[rv > 0]
    return rv


def build_har_features(rv):
    har = pd.DataFrame()
    har["RV"]   = rv
    har["RV_d"] = har["RV"].shift(1)
    har["RV_w"] = har["RV"].rolling(5).mean().shift(1)
    har["RV_m"] = har["RV"].rolling(22).mean().shift(1)
    har = har.dropna()
    return har


def fit_har(train, test):
    Y = train["RV"]
    X = sm.add_constant(train[["RV_d", "RV_w", "RV_m"]])
    model = sm.OLS(Y, X).fit()
    X_test = sm.add_constant(test[["RV_d", "RV_w", "RV_m"]], has_constant="add")
    forecast = model.predict(X_test).clip(lower=1e-10)
    forecast.index = test.index
    return forecast, model


def fit_garch(returns, train_end, test_index, n_minutes):
    garch_ret = returns * 100
    garch_train = garch_ret[garch_ret.index.date <= train_end.date()]
    spec = arch_model(garch_train, vol="Garch", p=1, q=1, dist="normal")
    fitted = spec.fit(disp="off")
    fc = fitted.forecast(horizon=TEST_DAYS, reindex=False)
    vars_ = fc.variance.values[-1, :] / 10000 * n_minutes
    forecast = pd.Series(vars_, index=test_index).clip(lower=1e-10)
    return forecast, fitted


def realized_kernel(returns_series, bandwidth=None):
    r = returns_series.values
    n = len(r)
    if bandwidth is None:
        bandwidth = max(1, int(np.ceil(n ** (1/3))))

    def parzen(x):
        ax = abs(x)
        if ax <= 0.5:
            return 1 - 6*ax**2 + 6*ax**3
        elif ax <= 1:
            return 2*(1 - ax)**3
        return 0.0

    total = np.sum(r**2)
    for h in range(1, bandwidth + 1):
        gamma_h = np.sum(r[h:] * r[:-h])
        total += 2 * parzen(h / bandwidth) * gamma_h
    return max(total, 1e-10)


def compute_rk_series(returns):
    rk_dict = {}
    for date, group in returns.groupby(returns.index.date):
        rk_dict[pd.Timestamp(date)] = realized_kernel(group)
    rk = pd.Series(rk_dict)
    return rk[rk > 0]


def build_har_rk_features(rv, rk):
    har_rk = pd.DataFrame()
    har_rk["RV"]   = rv
    har_rk["RK_d"] = rk.shift(1)
    har_rk["RK_w"] = rk.rolling(5).mean().shift(1)
    har_rk["RK_m"] = rk.rolling(22).mean().shift(1)
    har_rk = har_rk.dropna()
    return har_rk


def fit_har_rk(train_rk, test_rk):
    Y = train_rk["RV"]
    X = sm.add_constant(train_rk[["RK_d", "RK_w", "RK_m"]])
    model = sm.OLS(Y, X).fit()
    X_test = sm.add_constant(test_rk[["RK_d", "RK_w", "RK_m"]], has_constant="add")
    forecast = model.predict(X_test).clip(lower=1e-10)
    forecast.index = test_rk.index
    return forecast


def mse(actual, forecast):
    a = actual.values
    f = forecast.reindex(actual.index).values
    return float(np.mean((a - f)**2))


def qlike(actual, forecast):
    a = actual.values
    f = forecast.reindex(actual.index).values
    return float(np.mean(np.log(f) + a / f))



# MAIN LOOP


all_results = []
all_forecasts = {}

for ticker in TICKERS:
    print(f"\n{'='*50}")
    print(f"Processing: {ticker}")
    print(f"{'='*50}")

    try:
        # Data
        prices  = get_data(ticker)
        returns = compute_log_returns(prices)
        rv      = compute_rv(returns)
        n_min   = int(returns.groupby(returns.index.date).count().mean())

        # HAR features
        har  = build_har_features(rv)
        if len(har) < 25:
            print(f"  Skipping {ticker}: insufficient data ({len(har)} rows)")
            continue

        train = har.iloc[:-TEST_DAYS]
        test  = har.iloc[-TEST_DAYS:]

        # HAR-RV
        har_fc, _ = fit_har(train, test)

        # GARCH
        garch_fc, garch_fitted = fit_garch(
            returns, train.index[-1], test.index, n_min
        )

        # Realized Kernel
        rk     = compute_rk_series(returns)
        har_rk = build_har_rk_features(rv, rk)
        train_rk = har_rk.iloc[:-TEST_DAYS]
        test_rk  = har_rk.iloc[-TEST_DAYS:]
        har_rk_fc = fit_har_rk(train_rk, test_rk)

        # Align on common index
        idx = test.index
        realized = test["RV"]

        # Metrics
        row = {
            "Asset":         ticker,
            "GARCH_MSE":     mse(realized, garch_fc),
            "HAR_RV_MSE":    mse(realized, har_fc),
            "HAR_RK_MSE":    mse(realized, har_rk_fc),
            "GARCH_QLIKE":   qlike(realized, garch_fc),
            "HAR_RV_QLIKE":  qlike(realized, har_fc),
            "HAR_RK_QLIKE":  qlike(realized, har_rk_fc),
        }

        # Determine winners
        mse_vals   = {"GARCH": row["GARCH_MSE"],
                      "HAR-RV": row["HAR_RV_MSE"],
                      "HAR-RK": row["HAR_RK_MSE"]}
        qlike_vals = {"GARCH": row["GARCH_QLIKE"],
                      "HAR-RV": row["HAR_RV_QLIKE"],
                      "HAR-RK": row["HAR_RK_QLIKE"]}

        row["Winner_MSE"]   = min(mse_vals,   key=mse_vals.get)
        row["Winner_QLIKE"] = min(qlike_vals, key=qlike_vals.get)

        all_results.append(row)
        all_forecasts[ticker] = {
            "realized": realized,
            "GARCH":    garch_fc,
            "HAR-RV":   har_fc,
            "HAR-RK":   har_rk_fc
        }

        print(f"  MSE   — GARCH: {row['GARCH_MSE']:.3e} | "
              f"HAR-RV: {row['HAR_RV_MSE']:.3e} | "
              f"HAR-RK: {row['HAR_RK_MSE']:.3e} → Winner: {row['Winner_MSE']}")
        print(f"  QLIKE — GARCH: {row['GARCH_QLIKE']:.4f} | "
              f"HAR-RV: {row['HAR_RV_QLIKE']:.4f} | "
              f"HAR-RK: {row['HAR_RK_QLIKE']:.4f} → Winner: {row['Winner_QLIKE']}")

    except Exception as e:
        print(f"  ERROR on {ticker}: {e}")
        continue


# SUMMARY TABLE


results_df = pd.DataFrame(all_results).set_index("Asset")
print("\n" + "="*70)
print("FULL RESULTS SUMMARY")
print("="*70)
print(results_df.to_string())
results_df.to_csv("results/tables/multi_asset_results.csv")

# Win count
print("\n--- Model Win Count ---")
print("MSE wins:  ", results_df["Winner_MSE"].value_counts().to_dict())
print("QLIKE wins:", results_df["Winner_QLIKE"].value_counts().to_dict())


# PLOT: One subplot per asset


n_assets = len(all_forecasts)
fig, axes = plt.subplots(
    nrows=2, ncols=3,
    figsize=(16, 8),
    sharex=False
)
axes = axes.flatten()

for i, (ticker, data) in enumerate(all_forecasts.items()):
    ax = axes[i]
    ax.plot(data["realized"].index, data["realized"].values,
            label="Realized", color="black", linewidth=2, marker="o")
    ax.plot(data["GARCH"].index, data["GARCH"].values,
            label="GARCH", color="firebrick", linewidth=1.5,
            marker="^", linestyle="--")
    ax.plot(data["HAR-RV"].index, data["HAR-RV"].values,
            label="HAR-RV", color="steelblue", linewidth=1.5,
            marker="x", linestyle="--")
    ax.plot(data["HAR-RK"].index, data["HAR-RK"].values,
            label="HAR-RK", color="darkorange", linewidth=1.5,
            marker="s", linestyle="--")
    ax.set_title(ticker, fontsize=12, fontweight="bold")
    ax.set_ylabel("Realized Variance", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis="x", labelsize=7)
    if i == 0:
        ax.legend(fontsize=7)

# Hide unused subplots if any
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "Volatility Forecasts vs Realized Volatility\n"
    "5-min Intraday Data | Test Period: Apr 16-23, 2026",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig("results/figures/multi_asset_comparison.png", dpi=150)
plt.show()
print("\nPlot saved to results/figures/multi_asset_comparison.png")
print("Table saved to results/tables/multi_asset_results.csv")
