"""
Single-stock Value at Risk (VaR) and Expected Shortfall (ES) via
Filtered Historical Simulation (FHS), using EGARCH conditional volatility.

Method (computed walk-forward, once per day):
    1. Standardize each historical log return by that day's EGARCH
       conditional volatility forecast:  z_t = r_t / sigma_t
       This strips volatility clustering out of returns, leaving an
       (approximately) i.i.d. innovation series.
    2. For each day t, take the trailing 252-day window of standardized
       residuals *strictly before* t (z_{t-252} .. z_{t-1}) as the
       empirical innovation sample -- day t's own return is never used to
       forecast day t's risk, so this is a genuine out-of-sample backtest.
    3. Rescale those residuals by day t's EGARCH volatility forecast to
       build simulated one-day-ahead returns:
           r_sim_i = sigma_t * z_i,   i = 1..252
       This "filters in" day t's volatility regime while preserving the
       empirical shape (skew, fat tails) of realized shocks.
    4. VaR_t and ES_t are read off the empirical distribution of r_sim.
    5. r_t is then compared against -VaR_t to flag breaches, giving a
       rolling backtest of the model against realized returns.

Reference: Hull & White (1998) / Barone-Adesi, Giannopoulos & Vosper (1999)
filtered historical simulation.

Input: egarch_aapl_rolling_vol.csv, produced by egarch_aapl_rolling.py
(columns: Date, log_return, egarch_daily_vol, ...).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VOL_CSV = "egarch_aapl_rolling_vol.csv"
TICKER = "AAPL"

LOOKBACK = 252  # trading days of standardized residuals used as the empirical innovation sample
CONFIDENCE_LEVELS = [0.99]
PORTFOLIO_VALUE = 1_000_000  # notional for dollar VaR/ES; set to None to skip

# ---------------------------------------------------------------------------
# Load EGARCH volatility + returns
# ---------------------------------------------------------------------------
data = pd.read_csv(VOL_CSV, index_col="Date", parse_dates=True)
data = data[["log_return", "egarch_daily_vol"]].dropna()

if len(data) < LOOKBACK + 1:
    raise ValueError(f"Need at least {LOOKBACK + 1} days of data, have {len(data)}.")

# ---------------------------------------------------------------------------
# Standardized residuals: z_t = r_t / sigma_t
# ---------------------------------------------------------------------------
data["z"] = data["log_return"] / data["egarch_daily_vol"]


def var_es(returns, confidence):
    """Historical VaR/ES (as positive loss fractions) at the given confidence level."""
    alpha = 1 - confidence
    var_return = np.percentile(returns, alpha * 100)  # left-tail quantile of returns
    tail = returns[returns <= var_return]
    es_return = tail.mean() if len(tail) > 0 else var_return
    return -var_return, -es_return  # convert to positive loss fractions


# ---------------------------------------------------------------------------
# Walk-forward rolling VaR / ES
#
# For day t, use standardized residuals from [t-LOOKBACK, t-1] (excludes
# day t itself) rescaled by day t's EGARCH vol forecast -- so VaR_t/ES_t
# are known before r_t is observed.
# ---------------------------------------------------------------------------
returns = data["log_return"].to_numpy()
z = data["z"].to_numpy()
sigma = data["egarch_daily_vol"].to_numpy()
dates = data.index

records = []
for i in range(LOOKBACK, len(data)):
    window_z = z[i - LOOKBACK:i]
    simulated_returns = sigma[i] * window_z

    row = {"Date": dates[i], "realized_return": returns[i]}
    for cl in CONFIDENCE_LEVELS:
        var_pct, es_pct = var_es(simulated_returns, cl)
        row[f"VaR_{cl}"] = var_pct
        row[f"ES_{cl}"] = es_pct
    records.append(row)

rolling = pd.DataFrame(records).set_index("Date")

if PORTFOLIO_VALUE is not None:
    for cl in CONFIDENCE_LEVELS:
        rolling[f"VaR_{cl}_$"] = rolling[f"VaR_{cl}"] * PORTFOLIO_VALUE
        rolling[f"ES_{cl}_$"] = rolling[f"ES_{cl}"] * PORTFOLIO_VALUE

rolling.to_csv("single_stock_var_es_rolling.csv")

# ---------------------------------------------------------------------------
# Backtest summary: breach = realized loss worse than VaR threshold
# ---------------------------------------------------------------------------
print(f"{TICKER} - Filtered Historical Simulation VaR/ES (walk-forward)")
print(f"Lookback: {LOOKBACK} trading days  |  {len(rolling)} out-of-sample days\n")

for cl in CONFIDENCE_LEVELS:
    breaches = rolling["realized_return"] < -rolling[f"VaR_{cl}"]
    expected_rate = 1 - cl
    print(
        f"{cl:.0%} VaR: {breaches.sum()} breaches / {len(rolling)} days "
        f"({breaches.mean():.2%} observed vs {expected_rate:.2%} expected)"
    )

latest = rolling.iloc[-1]
print(f"\nLatest ({rolling.index[-1].date()}):")
print(latest.to_string(float_format=lambda x: f"{x:,.4f}"))

# ---------------------------------------------------------------------------
# Plot: VaR / ES vs realized returns, with breaches marked
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(rolling.index, rolling["realized_return"], color="steelblue", lw=0.8, label="Realized return")

colors = plt.cm.tab10.colors
for i, cl in enumerate(CONFIDENCE_LEVELS):
    color = colors[i % len(colors)]
    ax.plot(rolling.index, -rolling[f"VaR_{cl}"], color=color, lw=1.2, label=f"{cl:.0%} VaR (loss threshold)")
    ax.plot(rolling.index, -rolling[f"ES_{cl}"], color=color, lw=1.0, linestyle=":", label=f"{cl:.0%} ES")

    breach_mask = rolling["realized_return"] < -rolling[f"VaR_{cl}"]
    ax.scatter(
        rolling.index[breach_mask],
        rolling["realized_return"][breach_mask],
        color="red",
        s=18,
        zorder=5,
        label=f"{cl:.0%} VaR breach ({breach_mask.sum()})",
    )

ax.set_title(f"{TICKER} - Walk-forward FHS VaR/ES vs realized returns")
ax.set_ylabel("Log return")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
plt.show()
