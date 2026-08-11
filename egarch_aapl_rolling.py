"""
Rolling EGARCH(1,1) volatility model for Apple (AAPL).

Walk-forward scheme:
    - Training window : 504 trading days (~2 years)
    - Recalibration    : every 21 trading days (~1 month)

At each recalibration point the model is refit on the trailing 504-day
window of log returns, then used to forecast daily conditional volatility
for the next 21 (out-of-sample) trading days. Stitching these forecasts
together gives a continuous, walk-forward volatility series with no
look-ahead bias.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TICKER = "AAPL"
START_DATE = "2010-01-01"
END_DATE = None  # None -> up to today

TRAIN_WINDOW = 504  # trading days (~2 years)
RECAL_STEP = 21  # trading days (~1 month)

MEAN_MODEL = "Constant"
DIST = "t"  # Student's t: fatter tails than normal, standard choice for equities
P, O, Q = 1, 1, 1  # EGARCH(1,1)

TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
raw_data = yf.download(TICKER, start=START_DATE, end=END_DATE)
raw_data.columns = raw_data.columns.get_level_values(0)

prices = raw_data["Close"]
log_ret = np.log(prices / prices.shift(1)).dropna()

# arch converges more reliably on returns scaled to roughly O(1) magnitude
returns_pct = log_ret * 100

# ---------------------------------------------------------------------------
# Rolling walk-forward EGARCH fit / forecast
# ---------------------------------------------------------------------------
forecast_daily_vol = pd.Series(index=returns_pct.index, dtype=float)
window_params = []

n_obs = len(returns_pct)

for start in range(TRAIN_WINDOW, n_obs, RECAL_STEP):
    train = returns_pct.iloc[start - TRAIN_WINDOW:start]
    horizon = min(RECAL_STEP, n_obs - start)

    model = arch_model(
        train, mean=MEAN_MODEL, vol="EGARCH", p=P, o=O, q=Q, dist=DIST
    )
    res = model.fit(disp="off", options={"maxiter": 1000})

    # EGARCH has no closed-form multi-step forecast -> simulate.
    # Bootstrap resamples historical standardized residuals rather than
    # drawing from the fitted (sometimes very fat-tailed) parametric
    # distribution, which avoids occasional overflow in the exp() variance
    # recursion when a window's fit lands near the persistence boundary.
    fc = res.forecast(horizon=horizon, method="bootstrap", reindex=False)
    daily_vol_pct = np.sqrt(fc.variance.iloc[0].to_numpy())  # in % units

    # EGARCH models log-variance, so when a window's beta lands near the
    # persistence boundary (~unit root), compounding bootstrap shocks over
    # the horizon can blow up -- or underflow toward zero -- in exp()-space:
    # finite, but nonsense (e.g. 1000x the trailing realized vol, or a
    # near-zero forecast that later blows up any 1/vol calculation
    # downstream). Guard against non-finite values and both of these
    # "finite but absurd" cases by falling back to the window's
    # unconditional (long-run) volatility.
    sane_upper_bound_pct = 10 * train.std()
    sane_lower_bound_pct = 0.1 * train.std()
    if (
        not np.all(np.isfinite(daily_vol_pct))
        or np.any(daily_vol_pct > sane_upper_bound_pct)
        or np.any(daily_vol_pct < sane_lower_bound_pct)
    ):
        # Fall back to the trailing window's realized std rather than a
        # model-implied unconditional vol -- simple and always well-behaved.
        daily_vol_pct = np.full(horizon, train.std())
        print(f"  [!] unstable forecast at {train.index[-1].date()} (beta={res.params['beta[1]']:.4f}), using trailing realized vol fallback")

    target_dates = returns_pct.index[start:start + horizon]
    forecast_daily_vol.loc[target_dates] = daily_vol_pct / 100  # back to decimal

    window_params.append(
        {
            "fit_date": train.index[-1],
            "forecast_start": target_dates[0],
            "forecast_end": target_dates[-1],
            **res.params.to_dict(),
        }
    )

forecast_daily_vol = forecast_daily_vol.dropna()
forecast_annual_vol = forecast_daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

params_df = pd.DataFrame(window_params).set_index("fit_date")

# ---------------------------------------------------------------------------
# Realized volatility for comparison (21-day trailing realized std)
# ---------------------------------------------------------------------------
realized_daily_vol = log_ret.rolling(RECAL_STEP).std()
realized_annual_vol = realized_daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
out = pd.DataFrame(
    {
        "log_return": log_ret,
        "egarch_daily_vol": forecast_daily_vol,
        "egarch_annual_vol": forecast_annual_vol,
        "realized_21d_annual_vol": realized_annual_vol,
    }
).dropna(subset=["egarch_daily_vol"])

out.to_csv("egarch_aapl_rolling_vol.csv")
params_df.to_csv("egarch_aapl_rolling_params.csv")

print(out.tail())
print(f"\n{len(window_params)} recalibration windows fit.")
print(f"Latest window params ({window_params[-1]['fit_date'].date()}):")
print(params_df.iloc[-1])

plt.figure(figsize=(12, 5))
plt.plot(out.index, out["egarch_annual_vol"], label="EGARCH forecast (walk-forward)")
plt.plot(
    out.index,
    out["realized_21d_annual_vol"],
    label="Realized 21d vol",
    alpha=0.6,
)
plt.title(f"{TICKER} annualized volatility: rolling EGARCH(1,1) vs realized")
plt.ylabel("Annualized volatility")
plt.legend()
plt.tight_layout()
plt.show()
