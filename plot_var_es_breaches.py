"""
Plot the DCC-GARCH portfolio VaR / ES time series with backtest breaches
highlighted.

A breach is realized_pnl_1d < -VaR_1d_0.99 (matches the definition used in
Kupiec_Christofferson_Tests.py). Reads dcc_garch_portfolio_var_es.csv and
(optionally) dcc_garch_kupiec_christoffersen.csv for the summary annotation.

Output: var_es_breaches.png
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

INPUT_CSV = "dcc_garch_portfolio_var_es.csv"
BACKTEST_CSV = "dcc_garch_kupiec_christoffersen.csv"
OUTPUT_PNG = "var_es_breaches.png"
CONFIDENCE = 0.99

# ---------------------------------------------------------------------------
# Palette (validated categorical + status slots)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"    # VaR (slot 1)
ORANGE = "#eb6834"  # ES (slot 2)
CRITICAL = "#d03b3b"  # breach markers (status: critical)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data = pd.read_csv(INPUT_CSV, parse_dates=["Date"]).set_index("Date")

var_col = f"VaR_1d_{CONFIDENCE}"
es_col = f"ES_1d_{CONFIDENCE}"

realized_loss = -data["realized_pnl_1d"]
breach_mask = realized_loss > data[var_col]
breaches = data.loc[breach_mask]
breach_loss = realized_loss.loc[breach_mask]

n_breaches = int(breach_mask.sum())
n_obs = len(data)
observed_rate = n_breaches / n_obs
expected_rate = 1 - CONFIDENCE

try:
    backtest = pd.read_csv(BACKTEST_CSV)
    row = backtest.loc[backtest["confidence"] == CONFIDENCE].iloc[0]
    reject_cc = bool(row["reject_cc"])
except (FileNotFoundError, IndexError, KeyError):
    reject_cc = None

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6.5), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.fill_between(data.index, data[var_col], data[es_col], color=ORANGE, alpha=0.12, zorder=1)
ax.plot(data.index, data[es_col], color=ORANGE, linewidth=2, label=f"Expected Shortfall ({CONFIDENCE:.0%}, 1d)", zorder=2)
ax.plot(data.index, data[var_col], color=BLUE, linewidth=2, label=f"Value at Risk ({CONFIDENCE:.0%}, 1d)", zorder=3)
ax.scatter(
    breaches.index, breach_loss, color=CRITICAL, s=64, zorder=4,
    edgecolors=SURFACE, linewidths=1.2,
    label=f"Breach (realized loss > VaR): {n_breaches}",
)

# Chrome
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(BASELINE)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.tick_params(colors=INK_MUTED, labelsize=9)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax.set_ylabel("Loss ($)", color=INK_SECONDARY, fontsize=10)
ax.set_title(
    f"DCC-GARCH Portfolio VaR & ES ({CONFIDENCE:.0%}, 1-Day) — Backtest Breaches",
    color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=14,
)

legend = ax.legend(
    loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY,
)

summary = (
    f"{n_breaches} / {n_obs} breaches  "
    f"(observed {observed_rate:.2%} vs expected {expected_rate:.2%})"
)
if reject_cc is not None:
    summary += "\nConditional coverage: " + ("REJECTED (clustering)" if reject_cc else "not rejected")
ax.text(
    0.99, 0.98, summary, transform=ax.transAxes, ha="right", va="top",
    fontsize=9, color=INK_SECONDARY,
)

fig.tight_layout()
fig.savefig(OUTPUT_PNG, facecolor=SURFACE)
print(f"Saved {OUTPUT_PNG}  ({n_breaches} breaches out of {n_obs} observations)")
