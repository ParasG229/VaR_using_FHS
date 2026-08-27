"""
Rolling DCC-EGARCH Value-at-Risk / Expected Shortfall for a multi-asset
portfolio of stocks and options.

Walk-forward scheme (mirrors egarch_aapl_rolling.py / Single_Stock_VaR_Model.py,
extended to multiple correlated assets):

    Stage 1 - Univariate vol (per asset)
        Trailing 504-day (~2y) window, EGARCH(1,1)-t, refit every day,
        1-step-ahead analytic forecast sigma_{i,t}. Standardized residual
        z_{i,t} = r_{i,t} / sigma_{i,t} is out-of-sample (uses only the fit
        from data up to t-1), so it carries no look-ahead bias.

    Stage 2 - Dynamic correlation (DCC(1,1), Engle 2002)
        On the same trailing 504-day window of the z_{i,t} matrix, estimate
        scalar DCC parameters (a, b) by QMLE, refit every day:
            Q_t = (1-a-b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1}
            R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
        Qbar = sample covariance of z over the window. Forecast R_{t+1} is
        the one-step-ahead correlation matrix used for day t+1's simulation.

    Stage 3 - Multivariate filtered historical simulation (FHS)
        For each historical day s in the window, decorrelate that day's z_s
        by *that day's own* fitted correlation R_s (e_s = chol(R_s)^-1 z_s),
        giving a pool of near-i.i.d., unit-covariance innovations that still
        carry the empirical (non-normal, fat-tailed) shape of real shocks.
        To simulate day t+1: recolor pooled draws with tomorrow's forecast
        chol(R_{t+1}) and each asset's sigma_{i,t+1}, giving joint simulated
        returns that respect both the current vol regime and the current
        correlation regime.

        10-day horizon: block-bootstrap 10 consecutive days from the pool
        (preserves serial dependence) and recolor each day with the SAME
        frozen (R_{t+1}, sigma_{t+1}) -- i.e. the vol/correlation regime is
        held constant across the 10-day window rather than re-evolved
        day-by-day. This is a simplification (a true path-dependent 10-day
        simulation would re-run the EGARCH/DCC recursions along each
        simulated day) but is a standard, defensible FHS shortcut and avoids
        hand-reimplementing arch's internal EGARCH recursion.

    Stage 4 - Portfolio valuation
        Stock legs: linear P&L (qty * price change).
        Option legs: full Black-Scholes repricing on each simulated
        underlying path, holding implied vol at each contract's latest
        observed value ("sticky IV"), discounting with the supplied
        risk-free curve. No dividend yield curve in the input format (see
        DIVIDEND_YIELD below) -- flat assumption, refine later if needed.

        The current positions.csv snapshot is treated as a constant
        hypothetical portfolio walked backward through history (the
        standard way to backtest a snapshot portfolio's VaR model) -- dates
        after an option's expiry are skipped for that leg.

Raw data format (see Data/Templates/*.csv for examples):
    stock_prices.csv   : Date, Ticker, AdjClose
    option_prices.csv  : Date, OptionID, UnderlyingTicker, Type, Strike, Expiry, Price, ImpliedVol
    positions.csv      : InstrumentID, InstrumentType, Ticker, Quantity, Strike, Expiry, OptionType
    risk_free_rate.csv : Date, Rate
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from arch import arch_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "Data/Templates"
STOCK_PRICES_CSV = f"{DATA_DIR}/stock_prices.csv"
OPTION_PRICES_CSV = f"{DATA_DIR}/option_prices.csv"
POSITIONS_CSV = f"{DATA_DIR}/positions.csv"
RISK_FREE_CSV = f"{DATA_DIR}/risk_free_rate.csv"

START_DATE = None  # None -> use all available history; set e.g. "2022-01-01" for a faster test run

TRAIN_WINDOW = 504  # trading days (~2 years)
RECAL_STEP = 1  # recalibrate every day (both EGARCH and DCC)

MEAN_MODEL = "Constant"
DIST = "t"
P, O, Q = 1, 1, 1  # EGARCH(1,1)

DCC_BOUNDS = [(1e-6, 0.3), (1e-6, 0.995)]  # (a, b)
DCC_INIT = (0.03, 0.90)

CONFIDENCE_LEVELS = [0.99]
HORIZONS_DAYS = [1, 10]
N_SIMULATIONS = 5000
RANDOM_SEED = 42

DIVIDEND_YIELD = 0.0  # flat assumption; input format has no per-ticker dividend curve
TRADING_DAYS_PER_YEAR = 252

rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_stock_prices(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    wide = df.pivot(index="Date", columns="Ticker", values="AdjClose").sort_index()
    if START_DATE is not None:
        wide = wide.loc[START_DATE:]
    return wide


def load_option_prices(path):
    df = pd.read_csv(path, parse_dates=["Date", "Expiry"])
    df["Type"] = df["Type"].str.upper().str[0]  # normalize "Call"/"C"/"call" -> "C"
    return df


def load_positions(path):
    df = pd.read_csv(path, parse_dates=["Expiry"])
    df["InstrumentType"] = df["InstrumentType"].str.upper()
    df["OptionType"] = df["OptionType"].astype(str).str.upper().str[0]
    return df


def load_risk_free(path, index):
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df["Rate"].reindex(index).ffill().bfill()


def compute_log_returns(prices_wide):
    # Inner-join calendar: keep only dates where every asset has a price.
    # Fine for a set of equities/indices sharing one exchange calendar; if
    # you add assets with a different trading calendar (e.g. crypto), this
    # will silently drop the days they don't overlap.
    prices_wide = prices_wide.dropna(how="any")
    return np.log(prices_wide / prices_wide.shift(1)).dropna(how="any")


# ---------------------------------------------------------------------------
# Stage 1: rolling univariate EGARCH per asset
# ---------------------------------------------------------------------------
def rolling_univariate_egarch(returns_wide, train_window=TRAIN_WINDOW, recal_step=RECAL_STEP):
    """Walk-forward EGARCH(1,1)-t per column of returns_wide (in % units internally).

    Returns (vol_forecast, z), both DataFrames aligned to returns_wide.index,
    containing only the out-of-sample forecast period (index >= train_window).
    """
    tickers = returns_wide.columns
    n_obs = len(returns_wide)
    returns_pct = returns_wide * 100

    vol_forecast = pd.DataFrame(index=returns_wide.index, columns=tickers, dtype=float)

    for ticker in tickers:
        series = returns_pct[ticker]
        for start in range(train_window, n_obs, recal_step):
            train = series.iloc[start - train_window:start]
            horizon = min(recal_step, n_obs - start)

            model = arch_model(train, mean=MEAN_MODEL, vol="EGARCH", p=P, o=O, q=Q, dist=DIST)
            res = model.fit(disp="off", options={"maxiter": 500})

            method = "analytic" if horizon == 1 else "bootstrap"
            fc = res.forecast(horizon=horizon, method=method, reindex=False)
            daily_vol_pct = np.sqrt(fc.variance.iloc[0].to_numpy())

            sane_upper = 10 * train.std()
            sane_lower = 0.1 * train.std()
            if not np.all(np.isfinite(daily_vol_pct)) or np.any(daily_vol_pct > sane_upper) or np.any(daily_vol_pct < sane_lower):
                daily_vol_pct = np.full(horizon, train.std())

            target_dates = returns_pct.index[start:start + horizon]
            vol_forecast.loc[target_dates, ticker] = daily_vol_pct / 100

        print(f"  [univariate EGARCH] {ticker} done")

    vol_forecast = vol_forecast.dropna(how="any")
    z = (returns_wide.loc[vol_forecast.index] / vol_forecast).dropna(how="any")
    vol_forecast = vol_forecast.loc[z.index]
    return vol_forecast, z


# ---------------------------------------------------------------------------
# Stage 2: DCC(1,1) - scalar, two-step QMLE
# ---------------------------------------------------------------------------
def dcc_filter(a, b, z, Qbar):
    """Run the DCC(1,1) recursion. z: (T, n) array. Returns R_series (T, n, n) and Q_next (n, n)."""
    T, n = z.shape
    Q_t = Qbar.copy()
    R_series = np.empty((T, n, n))
    for t in range(T):
        d = np.sqrt(np.diag(Q_t))
        R_series[t] = Q_t / np.outer(d, d)
        zt = z[t]
        Q_t = (1 - a - b) * Qbar + a * np.outer(zt, zt) + b * Q_t
    return R_series, Q_t  # Q_t here is the one-step-ahead state for T (i.e. forecast input for T+1)


def dcc_negloglik(params, z, Qbar):
    a, b = params
    if a < 0 or b < 0 or a + b >= 1:
        return 1e10
    R_series, _ = dcc_filter(a, b, z, Qbar)
    T = z.shape[0]
    nll = 0.0
    for t in range(T):
        R = R_series[t]
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            return 1e10
        zt = z[t]
        nll += logdet + zt @ np.linalg.solve(R, zt) - zt @ zt
    return 0.5 * nll


def fit_dcc(z_window, init=DCC_INIT):
    Qbar = np.cov(z_window.T)
    res = minimize(
        dcc_negloglik, x0=init, args=(z_window, Qbar),
        method="L-BFGS-B", bounds=DCC_BOUNDS, options={"maxiter": 200},
    )
    a, b = res.x
    R_series, Q_next = dcc_filter(a, b, z_window, Qbar)

    # One-step-ahead forecast correlation matrix for "tomorrow"
    d = np.sqrt(np.diag(Q_next))
    R_forecast = Q_next / np.outer(d, d)

    return a, b, R_series, R_forecast


# ---------------------------------------------------------------------------
# Stage 3: multivariate FHS simulation
# ---------------------------------------------------------------------------
def build_innovation_pool(z_window, R_series):
    """Decorrelate each historical day's z by that day's own fitted R -> pooled innovations."""
    T, n = z_window.shape
    pool = np.empty((T, n))
    for t in range(T):
        L = np.linalg.cholesky(R_series[t])
        pool[t] = np.linalg.solve(L, z_window[t])
    return pool


def simulate_returns(pool, sigma_next, R_forecast, horizon_days, n_sims=N_SIMULATIONS):
    """Simulate n_sims joint horizon-day log-return paths (summed over horizon).

    Recolors pooled innovations with the forecast correlation/vol; for
    horizon_days > 1, block-bootstraps consecutive days from the pool and
    holds (R_forecast, sigma_next) frozen across the block (see module
    docstring, Stage 3).
    """
    T_pool, n = pool.shape
    L_forecast = np.linalg.cholesky(R_forecast)

    total_log_return = np.zeros((n_sims, n))
    max_start = T_pool - horizon_days
    for _ in range(horizon_days):
        idx = rng.integers(0, max(max_start, 1), size=n_sims)
        e = pool[idx]  # (n_sims, n)
        z_sim = e @ L_forecast.T
        total_log_return += z_sim * sigma_next
    return total_log_return  # (n_sims, n), sum of daily log returns over the horizon


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------
def bs_price(S, K, T, r, sigma, option_type, q=DIVIDEND_YIELD):
    S = np.asarray(S, dtype=float)
    T = max(T, 1e-6)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if option_type == "C":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    else:
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Portfolio P&L from simulated underlying returns
# ---------------------------------------------------------------------------
def latest_iv(option_prices, option_id, as_of_date):
    hist = option_prices[(option_prices["OptionID"] == option_id) & (option_prices["Date"] <= as_of_date)]
    if hist.empty:
        return None
    return hist.sort_values("Date")["ImpliedVol"].iloc[-1]


def portfolio_pnl(sim_log_returns, tickers, spot_today, positions, option_prices, as_of_date, horizon_days, r):
    """sim_log_returns: (n_sims, n) aligned to `tickers`. Returns array of portfolio P&L per sim."""
    sim_prices = spot_today.values * np.exp(sim_log_returns)  # (n_sims, n)
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    pnl = np.zeros(sim_log_returns.shape[0])

    for _, pos in positions.iterrows():
        if pos["InstrumentType"] == "STOCK":
            if pos["Ticker"] not in ticker_idx:
                continue
            i = ticker_idx[pos["Ticker"]]
            s0 = spot_today[pos["Ticker"]]
            pnl += pos["Quantity"] * (sim_prices[:, i] - s0)

        elif pos["InstrumentType"] == "OPTION":
            if pos["Ticker"] not in ticker_idx:
                continue
            expiry = pos["Expiry"]
            t_today = (expiry - as_of_date).days / 365.0
            t_horizon = (expiry - (as_of_date + pd.Timedelta(days=horizon_days))).days / 365.0
            if t_today <= 0:
                continue  # expired as of this backtest date -- excluded from that day's portfolio

            iv = latest_iv(option_prices, pos["InstrumentID"], as_of_date)
            if iv is None or not np.isfinite(iv):
                continue

            i = ticker_idx[pos["Ticker"]]
            s0 = spot_today[pos["Ticker"]]
            price_today = bs_price(s0, pos["Strike"], t_today, r, iv, pos["OptionType"])
            price_sim = bs_price(sim_prices[:, i], pos["Strike"], t_horizon, r, iv, pos["OptionType"])
            pnl += pos["Quantity"] * (price_sim - price_today)

    return pnl


def var_es(pnl, confidence):
    alpha = 1 - confidence
    var_loss = -np.percentile(pnl, alpha * 100)
    tail = pnl[pnl <= -var_loss]
    es_loss = -tail.mean() if len(tail) > 0 else var_loss
    return var_loss, es_loss


# ---------------------------------------------------------------------------
# Main walk-forward loop
# ---------------------------------------------------------------------------
def main():
    stock_prices = load_stock_prices(STOCK_PRICES_CSV)
    option_prices = load_option_prices(OPTION_PRICES_CSV)
    positions = load_positions(POSITIONS_CSV)
    log_ret = compute_log_returns(stock_prices)
    risk_free = load_risk_free(RISK_FREE_CSV, log_ret.index)

    print(f"Assets: {list(log_ret.columns)}  |  {len(log_ret)} return observations")

    print("Stage 1: rolling univariate EGARCH...")
    vol_forecast, z = rolling_univariate_egarch(log_ret)
    print(f"  -> {len(z)} days of out-of-sample vol/z forecasts")

    tickers = list(z.columns)
    n_obs = len(z)

    records = []
    a_prev, b_prev = DCC_INIT

    for start in range(TRAIN_WINDOW, n_obs, RECAL_STEP):
        z_window = z.iloc[start - TRAIN_WINDOW:start].to_numpy()
        as_of_date = z.index[start - 1]
        forecast_date = z.index[start]

        a, b, R_series, R_forecast = fit_dcc(z_window, init=(a_prev, b_prev))
        a_prev, b_prev = a, b

        pool = build_innovation_pool(z_window, R_series)
        sigma_next = vol_forecast.loc[forecast_date, tickers].to_numpy()
        spot_today = stock_prices.loc[as_of_date, tickers]
        r = risk_free.loc[as_of_date]

        row = {"Date": forecast_date, "dcc_a": a, "dcc_b": b}
        for h in HORIZONS_DAYS:
            sim_returns = simulate_returns(pool, sigma_next, R_forecast, h)
            pnl = portfolio_pnl(sim_returns, tickers, spot_today, positions, option_prices, as_of_date, h, r)
            for cl in CONFIDENCE_LEVELS:
                var_loss, es_loss = var_es(pnl, cl)
                row[f"VaR_{h}d_{cl}"] = var_loss
                row[f"ES_{h}d_{cl}"] = es_loss

        # Realized 1-day portfolio P&L (actual next-day return, not simulated) --
        # the "ground truth" outcome that Kupiec/Christoffersen breach tests
        # compare the 1-day VaR forecast against. Only 1-day is recorded: the
        # 10-day VaR uses overlapping windows, which violates the iid breach
        # assumption those tests rely on.
        realized_return = log_ret.loc[forecast_date, tickers].to_numpy().reshape(1, -1)
        realized_pnl = portfolio_pnl(
            realized_return, tickers, spot_today, positions, option_prices, as_of_date, 1, r
        )[0]
        row["realized_pnl_1d"] = realized_pnl

        records.append(row)

        if len(records) % 50 == 0:
            print(f"  [{len(records)}] {forecast_date.date()}  a={a:.4f} b={b:.4f}")

    results = pd.DataFrame(records).set_index("Date")
    results.to_csv("dcc_garch_portfolio_var_es.csv")

    print(f"\n{len(results)} recalibration days computed.")
    print(results.tail())


if __name__ == "__main__":
    main()
