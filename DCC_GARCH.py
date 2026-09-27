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
        On a trailing 504-day window of the z_{i,t} matrix -- itself only
        available from day 504 of the *out-of-sample* Stage 1 output, so
        this stage's first estimate needs roughly 2*TRAIN_WINDOW (~1008,
        ~4 years) trading days of underlying return history in total, not
        just TRAIN_WINDOW. main() raises a clear error if there isn't
        enough history rather than silently producing zero output rows.
        Estimate scalar DCC parameters (a, b) by QMLE, refit every day:
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

        Option legs (OPTION_ROLL only): a rolled constant-maturity synthetic
        option (e.g. "always the ~30-day ATM call"), described by
        TargetTenorDays/TargetMoneyness instead of a fixed contract.
        Reconstructed each day from the implied-vol surface in
        vol_surface.csv (fixed tenor/moneyness nodes -- e.g. a vendor's own
        vol surface rather than raw per-contract quotes) by interpolating
        implied vol at the target tenor/moneyness: linear across moneyness
        within each surface tenor node, then linear in total variance across
        the two bracketing tenor nodes (see surface_iv()) -- has data across
        the full backtest window as long as the surface does, since it isn't
        tied to one contract's finite life. Repriced via full Black-Scholes
        on each simulated underlying path, discounting with the supplied
        risk-free curve. No dividend yield curve in the input format (see
        DIVIDEND_YIELD below) -- flat assumption, refine later if needed.

        The current positions.csv snapshot is treated as a constant
        hypothetical portfolio walked backward through history (the
        standard way to backtest a snapshot portfolio's VaR model).  Legs
        that can't be priced on a given day (surface doesn't bracket the
        target tenor) are excluded from that day's P&L and recorded in the
        excluded_legs output column -- not silently dropped.

Raw data format (see Data/Templates/*.csv for examples):
    stock_prices.csv   : Date, Ticker, AdjClose
    vol_surface.csv     : Date, UnderlyingTicker, TenorDays, Moneyness, ImpliedVol
                          (a fixed tenor/moneyness implied-vol surface -- e.g. a vendor's own
                          vol surface rather than raw per-contract quotes; Moneyness is
                          Strike/Spot - 1, so 0.0 = ATM, -0.10/+0.10 = 10% below/above spot)
    positions.csv       : InstrumentID, InstrumentType, Ticker, Quantity,
                           OptionType, TargetTenorDays, TargetMoneyness    (OPTION_ROLL rows)
    risk_free_rate.csv : Date, TenorDays, Rate
                          (a zero-rate curve -- multiple tenor nodes per date, e.g. O/N
                          through 1Y SOFR OIS or Treasury CMT. Each option leg is discounted
                          at the rate interpolated to *its own* time-to-expiry rather than one
                          flat rate for the whole portfolio -- see risk_free_rate())
"""

import bisect

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from arch import arch_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "Data/Processed"
STOCK_PRICES_CSV = f"{DATA_DIR}/stock_prices.csv"
VOL_SURFACE_CSV = f"{DATA_DIR}/vol_surface.csv"
POSITIONS_CSV = f"{DATA_DIR}/positions.csv"
RISK_FREE_CSV = f"{DATA_DIR}/risk_free_rate.csv"

START_DATE = None  # None -> use all available history; set e.g. "2022-01-01" for a faster test run

TRAIN_WINDOW = 252  # trading days (~2 years)
RECAL_STEP = 1  # recalibrate every day (both EGARCH and DCC)

MEAN_MODEL = "Constant"
DIST = "t"
P, O, Q = 1, 1, 1  # EGARCH(1,1)

DCC_BOUNDS = [(1e-6, 0.3), (1e-6, 0.995)]  # (a, b)
DCC_INIT = (0.03, 0.90)

CONFIDENCE_LEVELS = [0.99]
HORIZONS_DAYS = [1, 10]
N_SIMULATIONS = 10000
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


def load_vol_surface(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df["UnderlyingTicker"] = df["UnderlyingTicker"].str.upper()
    df["TenorDays"] = pd.to_numeric(df["TenorDays"])
    df["Moneyness"] = pd.to_numeric(df["Moneyness"])
    return df


def load_positions(path):
    df = pd.read_csv(path)
    df["InstrumentType"] = df["InstrumentType"].str.upper()
    df["OptionType"] = df["OptionType"].astype(str).str.upper().str[0]
    df["TargetTenorDays"] = pd.to_numeric(df["TargetTenorDays"], errors="coerce")
    df["TargetMoneyness"] = pd.to_numeric(df["TargetMoneyness"], errors="coerce")
    return df


def load_risk_free_curve(path):
    """Zero-rate curve, multiple tenor nodes per date (see module docstring). Returns
    (curve_by_date, available_dates) for risk_free_rate() -- available_dates is sorted so
    that function can do a sticky "latest curve on or before as_of_date" lookup."""
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values(["Date", "TenorDays"])
    curve_by_date = {
        date: (grp["TenorDays"].to_numpy(dtype=float), grp["Rate"].to_numpy(dtype=float))
        for date, grp in df.groupby("Date")
    }
    available_dates = sorted(curve_by_date.keys())
    return curve_by_date, available_dates


def risk_free_rate(rf_curve, as_of_date, tenor_days):
    """Zero rate for `tenor_days` maturity from the risk-free curve as of as_of_date --
    each option leg is discounted at the rate matched to *its own* time-to-expiry rather
    than one flat rate for the whole portfolio (see module docstring). Sticky: uses the
    latest published curve on or before as_of_date. Interpolated linearly in the
    compounding exponent (rate * tenor_days) between the two bracketing curve tenors --
    the same total-variance-style approach as surface_iv()'s tenor interpolation -- and
    flat-extrapolated beyond the curve's own tenor range (short-rate curves aren't
    reliably extrapolated in log space the way an implied-vol surface is)."""
    curve_by_date, available_dates = rf_curve
    pos = bisect.bisect_right(available_dates, as_of_date) - 1
    if pos < 0:
        raise ValueError(f"No risk-free curve available on or before {as_of_date.date()}")
    tenors, rates = curve_by_date[available_dates[pos]]

    tenor_days = max(tenor_days, 1e-6)
    if tenor_days <= tenors[0]:
        return float(rates[0])
    if tenor_days >= tenors[-1]:
        return float(rates[-1])

    total = rates * tenors
    idx = np.searchsorted(tenors, tenor_days) - 1
    t0, t1 = tenors[idx], tenors[idx + 1]
    v0, v1 = total[idx], total[idx + 1]
    w = (tenor_days - t0) / (t1 - t0)
    return float((v0 + w * (v1 - v0)) / tenor_days)


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
def surface_iv(vol_surface, underlying, as_of_date, target_tenor_days, target_moneyness):
    """Interpolate an implied vol at (target_tenor_days, target_moneyness) from a fixed
    tenor/moneyness vol surface (vol_surface.csv) quoted on as_of_date, for a rolled
    constant-maturity synthetic option position.

    Method: within each surface tenor node present that day, interpolate IV across
    moneyness linearly to get that node's IV at target_moneyness; then interpolate across
    the two bracketing tenor nodes linearly in total variance (IV^2 * tenor_days) -- the
    same approach the CBOE uses to blend near/far contracts into a constant-maturity level.
    Unlike a raw per-contract chain, the surface is already expressed in moneyness terms
    and isn't split by option type (put/call implied vol at a given strike is identical
    under Black-Scholes), so no spot or option-type argument is needed here.

    Returns (iv, note): note is None on a clean two-sided bracket, or a short string flagging
    a fallback (single tenor node, or target tenor outside the available nodes) so the caller
    can still price the leg but surface the caveat rather than silently absorbing it.
    """
    day_surf = vol_surface[
        (vol_surface["UnderlyingTicker"] == underlying) & (vol_surface["Date"] == as_of_date)
    ]
    if day_surf.empty:
        return None, "no surface quotes for this date"

    tenor_points = []
    for tenor, grp in day_surf.groupby("TenorDays"):
        grp = grp.sort_values("Moneyness")
        if grp["Moneyness"].nunique() >= 2:
            iv_at_target = np.interp(target_moneyness, grp["Moneyness"], grp["ImpliedVol"])
        else:
            iv_at_target = grp["ImpliedVol"].iloc[(grp["Moneyness"] - target_moneyness).abs().argmin()]
        tenor_points.append((tenor, iv_at_target))

    tenor_points.sort()
    tenors = np.array([t for t, _ in tenor_points], dtype=float)
    ivs = np.array([iv for _, iv in tenor_points])

    if len(tenor_points) == 1:
        return float(ivs[0]), "only one tenor node available -- no tenor interpolation"

    total_var = ivs ** 2 * tenors
    if target_tenor_days in tenors:
        return float(ivs[tenors == target_tenor_days][0]), None
    elif target_tenor_days < tenors[0]:
        note, idx = "target tenor before nearest available surface node -- extrapolated", 0
    elif target_tenor_days > tenors[-1]:
        note, idx = "target tenor beyond farthest available surface node -- extrapolated", len(tenors) - 2
    else:
        note, idx = None, np.searchsorted(tenors, target_tenor_days) - 1

    t0, t1 = tenors[idx], tenors[idx + 1]
    v0, v1 = total_var[idx], total_var[idx + 1]
    w = (target_tenor_days - t0) / (t1 - t0)
    var_target = v0 + w * (v1 - v0)
    iv_target = np.sqrt(max(var_target, 1e-8) / target_tenor_days)
    return float(iv_target), note


def portfolio_pnl(sim_log_returns, tickers, spot_today, positions, vol_surface, as_of_date, horizon_days, rf_curve):
    """sim_log_returns: (n_sims, n) aligned to `tickers`. Returns (pnl, excluded) where
    excluded is a list of (InstrumentID, reason) for legs dropped from this day's portfolio
    -- e.g. an option contract that hadn't started trading yet this far back in the backtest,
    or had already expired. Dropped legs are surfaced, not silently absorbed, so the reported
    VaR/ES for a given day is clearly labeled as to which legs it actually reflects.

    Each option leg is discounted at the risk-free rate for *its own* time-to-expiry
    (via risk_free_rate()), not one flat portfolio-wide rate -- see module docstring.
    """
    sim_prices = spot_today.values * np.exp(sim_log_returns)  # (n_sims, n)
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    pnl = np.zeros(sim_log_returns.shape[0])
    excluded = []

    for _, pos in positions.iterrows():
        if pos["InstrumentType"] == "STOCK":
            if pos["Ticker"] not in ticker_idx:
                excluded.append((pos["InstrumentID"], "ticker not in return series"))
                continue
            i = ticker_idx[pos["Ticker"]]
            s0 = spot_today[pos["Ticker"]]
            pnl += pos["Quantity"] * (sim_prices[:, i] - s0)

        elif pos["InstrumentType"] == "OPTION_ROLL":
            # Rolled constant-maturity synthetic option (e.g. "always the ~30-day ATM
            # call"), reconstructed from the implied-vol surface each day rather than
            # tracking one decaying contract -- see surface_iv(). Tenor is held constant
            # across the simulation horizon (it's rebalanced back to target daily, not
            # decaying) and the strike is rebalanced to spot*(1+target_moneyness) on each
            # simulated path, tracking constant relative moneyness the same way the "today"
            # leg does. Constant tenor -> same discount rate for both legs.
            if pos["Ticker"] not in ticker_idx:
                excluded.append((pos["InstrumentID"], "underlying ticker not in return series"))
                continue
            i = ticker_idx[pos["Ticker"]]
            s0 = spot_today[pos["Ticker"]]
            target_tenor = pos["TargetTenorDays"]
            target_moneyness = pos["TargetMoneyness"]

            iv, note = surface_iv(vol_surface, pos["Ticker"], as_of_date, target_tenor, target_moneyness)
            if iv is None:
                excluded.append((pos["InstrumentID"], note))
                continue
            if note:
                excluded.append((pos["InstrumentID"], f"included with caveat: {note}"))

            t_target = target_tenor / 365.0
            r_target = risk_free_rate(rf_curve, as_of_date, target_tenor)
            strike_today = s0 * (1 + target_moneyness)
            price_today = bs_price(s0, strike_today, t_target, r_target, iv, pos["OptionType"])

            strike_sim = sim_prices[:, i] * (1 + target_moneyness)
            price_sim = bs_price(sim_prices[:, i], strike_sim, t_target, r_target, iv, pos["OptionType"])
            pnl += pos["Quantity"] * (price_sim - price_today)

    return pnl, excluded


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
    vol_surface = load_vol_surface(VOL_SURFACE_CSV)
    positions = load_positions(POSITIONS_CSV)
    log_ret = compute_log_returns(stock_prices)
    rf_curve = load_risk_free_curve(RISK_FREE_CSV)

    print(f"Assets: {list(log_ret.columns)}  |  {len(log_ret)} return observations")

    print("Stage 1: rolling univariate EGARCH...")
    vol_forecast, z = rolling_univariate_egarch(log_ret)
    print(f"  -> {len(z)} days of out-of-sample vol/z forecasts")

    tickers = list(z.columns)
    n_obs = len(z)

    # z already burned TRAIN_WINDOW days of returns for the EGARCH stage (Stage 1);
    # the DCC stage (Stage 2) then needs its own trailing TRAIN_WINDOW window of z,
    # so the model needs roughly 2*TRAIN_WINDOW trading days of return history in
    # total before it can produce a single output row.
    if n_obs <= TRAIN_WINDOW:
        raise ValueError(
            f"Only {n_obs} days of out-of-sample vol/z forecasts available, but "
            f"TRAIN_WINDOW={TRAIN_WINDOW} days are needed for the DCC stage on top "
            f"of that -- need roughly 2*TRAIN_WINDOW ({2 * TRAIN_WINDOW}) trading "
            f"days of return history in total (~{2 * TRAIN_WINDOW / TRADING_DAYS_PER_YEAR:.1f} "
            f"years). Provide more history, an earlier START_DATE, or a smaller TRAIN_WINDOW."
        )

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

        row = {"Date": forecast_date, "dcc_a": a, "dcc_b": b}
        excluded_all = []
        for h in HORIZONS_DAYS:
            sim_returns = simulate_returns(pool, sigma_next, R_forecast, h)
            pnl, excluded = portfolio_pnl(sim_returns, tickers, spot_today, positions, vol_surface, as_of_date, h, rf_curve)
            excluded_all.extend(excluded)  # same exclusion set each horizon (expiry/IV checks don't depend on h)
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
        realized_pnl_arr, _ = portfolio_pnl(
            realized_return, tickers, spot_today, positions, vol_surface, as_of_date, 1, rf_curve
        )
        row["realized_pnl_1d"] = realized_pnl_arr[0]

        excluded_unique = sorted(set(excluded_all))
        row["excluded_legs"] = "; ".join(f"{inst_id} ({reason})" for inst_id, reason in excluded_unique)

        records.append(row)

        if len(records) % 50 == 0:
            print(f"  [{len(records)}] {forecast_date.date()}  a={a:.4f} b={b:.4f}")

    results = pd.DataFrame(records).set_index("Date")
    results.to_csv("dcc_garch_portfolio_var_es.csv")

    print(f"\n{len(results)} recalibration days computed.")
    print(results.tail())


if __name__ == "__main__":
    main()
