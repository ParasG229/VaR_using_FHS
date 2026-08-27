"""
Kupiec (unconditional coverage) and Christoffersen (independence /
conditional coverage) backtests for the walk-forward VaR model.

Given a series of VaR breach indicators I_t = 1{realized_return_t < -VaR_t},
three likelihood-ratio tests are run:

    1. Kupiec POF (proportion of failures) test
       H0: the observed breach rate equals the expected exception rate
       (1 - confidence). Tests only whether the *number* of breaches is
       right, not when they happen. LR_uc ~ chi2(1) under H0.

    2. Christoffersen independence test
       H0: breaches are independent over time (no clustering), by
       comparing a 2-state Markov chain fit on the breach sequence
       against the null of a constant breach probability.
       LR_ind ~ chi2(1) under H0.

    3. Christoffersen conditional coverage test (combined)
       H0: both correct unconditional coverage AND independence.
       LR_cc = LR_uc + LR_ind ~ chi2(2) under H0.

A good VaR model should fail to reject all three: right number of
breaches, and breaches that aren't clustered (clustering signals the
model is slow to react to changing volatility).

Reference: Kupiec (1995); Christoffersen (1998).

Input: a walk-forward VaR backtest CSV indexed by Date, with a realized-outcome
column and one or more VaR columns. Two producers are supported:

    single_stock_var_es_rolling.csv (Single_Stock_VaR_Model.py)
        columns: Date, realized_return, VaR_<confidence>, ...
        (VaR is in return units)

    dcc_garch_portfolio_var_es.csv (DCC_GARCH.py)
        columns: Date, realized_pnl_1d, VaR_<horizon>d_<confidence>, ...
        (VaR is in dollar P&L units)

Only 1-day-horizon VaR columns are testable here: the LR tests below assume
an iid daily breach sequence, and a multi-day VaR's realized outcome would
need overlapping windows, which manufactures spurious clustering regardless
of model quality. Any non-1-day VaR column found in the input is skipped.
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROLLING_CSV = "single_stock_var_es_rolling.csv"
OUTPUT_CSV = "var_backtest_kupiec_christoffersen.csv"  # change alongside ROLLING_CSV to avoid overwriting other backtests
LABEL = "AAPL"  # display label only (a ticker for single-stock, a portfolio name otherwise)
SIGNIFICANCE = 0.05  # reject H0 (model misspecified) if p-value < this

# First of these found in the data is used as the realized 1-day outcome.
REALIZED_COL_CANDIDATES = ["realized_return", "realized_pnl_1d", "realized_pnl"]

# ---------------------------------------------------------------------------
# Load walk-forward VaR backtest data
# ---------------------------------------------------------------------------
data = pd.read_csv(ROLLING_CSV, index_col="Date", parse_dates=True)

VAR_COL_RE = re.compile(r"^VaR_(?:(\d+)d_)?([0-9.]+)$")  # VaR_0.99  or  VaR_10d_0.99

var_cols_1d = {}  # confidence -> column name
for c in data.columns:
    m = VAR_COL_RE.match(c)
    if not m:
        continue
    horizon = int(m.group(1)) if m.group(1) else 1
    confidence = float(m.group(2))
    if horizon != 1:
        print(f"Skipping {c}: {horizon}-day horizon isn't testable with this iid breach framework "
              f"(see module docstring).")
        continue
    var_cols_1d[confidence] = c

if not var_cols_1d:
    raise ValueError(f"No 1-day VaR_<confidence> columns found in {ROLLING_CSV}.")

realized_col = next((c for c in REALIZED_COL_CANDIDATES if c in data.columns), None)
if realized_col is None:
    raise ValueError(f"No realized-outcome column found in {ROLLING_CSV} (looked for {REALIZED_COL_CANDIDATES}).")

confidence_levels = sorted(var_cols_1d)


def kupiec_pof_test(breaches, confidence):
    """Unconditional coverage LR test. Returns (LR_uc, p_value, breach_rate)."""
    n = len(breaches)
    x = int(breaches.sum())
    p = 1 - confidence  # expected exception rate
    pi_hat = x / n

    log_l_null = x * np.log(p) + (n - x) * np.log(1 - p) if 0 < p < 1 else 0.0
    if x == 0 or x == n:
        log_l_alt = 0.0  # pi_hat is 0 or 1 -> likelihood at MLE is 1
    else:
        log_l_alt = x * np.log(pi_hat) + (n - x) * np.log(1 - pi_hat)

    lr_uc = -2 * (log_l_null - log_l_alt)
    p_value = 1 - chi2.cdf(lr_uc, df=1)
    return lr_uc, p_value, pi_hat


def christoffersen_independence_test(breaches):
    """Independence LR test on the breach sequence. Returns (LR_ind, p_value)."""
    prev = breaches[:-1]
    curr = breaches[1:]

    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    n0, n1 = n00 + n01, n10 + n11
    pi = (n01 + n11) / (n0 + n1)
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0

    def _log_term(prob, count):
        return count * np.log(prob) if count > 0 and 0 < prob < 1 else 0.0

    log_l_null = _log_term(pi, n01 + n11) + _log_term(1 - pi, n00 + n10)
    log_l_alt = (
        _log_term(pi01, n01) + _log_term(1 - pi01, n00)
        + _log_term(pi11, n11) + _log_term(1 - pi11, n10)
    )

    lr_ind = -2 * (log_l_null - log_l_alt)
    p_value = 1 - chi2.cdf(lr_ind, df=1)
    return lr_ind, p_value


# ---------------------------------------------------------------------------
# Run tests per confidence level
# ---------------------------------------------------------------------------
print(f"{LABEL} - VaR backtest: Kupiec & Christoffersen tests")
print(f"Realized outcome column: {realized_col}  |  {len(data)} out-of-sample days  |  "
      f"significance level: {SIGNIFICANCE:.0%}\n")

results = []
for cl in confidence_levels:
    var_col = var_cols_1d[cl]
    breaches = (data[realized_col] < -data[var_col]).to_numpy().astype(int)

    lr_uc, p_uc, breach_rate = kupiec_pof_test(breaches, cl)
    lr_ind, p_ind = christoffersen_independence_test(breaches)
    lr_cc = lr_uc + lr_ind
    p_cc = 1 - chi2.cdf(lr_cc, df=2)

    results.append({
        "confidence": cl,
        "n_breaches": int(breaches.sum()),
        "n_obs": len(breaches),
        "expected_rate": 1 - cl,
        "observed_rate": breach_rate,
        "LR_uc": lr_uc,
        "p_uc": p_uc,
        "reject_uc": p_uc < SIGNIFICANCE,
        "LR_ind": lr_ind,
        "p_ind": p_ind,
        "reject_ind": p_ind < SIGNIFICANCE,
        "LR_cc": lr_cc,
        "p_cc": p_cc,
        "reject_cc": p_cc < SIGNIFICANCE,
    })

    print(f"--- {cl:.0%} VaR ---")
    print(f"  Breaches: {breaches.sum()} / {len(breaches)}  "
          f"(observed {breach_rate:.2%} vs expected {1 - cl:.2%})")
    print(f"  Kupiec POF (unconditional coverage):   "
          f"LR={lr_uc:6.3f}  p={p_uc:.4f}  "
          f"{'REJECT H0 (bad coverage)' if p_uc < SIGNIFICANCE else 'fail to reject (OK)'}")
    print(f"  Christoffersen independence:           "
          f"LR={lr_ind:6.3f}  p={p_ind:.4f}  "
          f"{'REJECT H0 (clustered breaches)' if p_ind < SIGNIFICANCE else 'fail to reject (OK)'}")
    print(f"  Christoffersen conditional coverage:   "
          f"LR={lr_cc:6.3f}  p={p_cc:.4f}  "
          f"{'REJECT H0 (misspecified)' if p_cc < SIGNIFICANCE else 'fail to reject (OK)'}")
    print()

results_df = pd.DataFrame(results).set_index("confidence")
results_df.to_csv(OUTPUT_CSV)
