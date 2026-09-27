"""
Reshapes the raw Bloomberg-exported workbooks in Data/*.xlsx into the 5 CSVs
DCC_GARCH.py expects (see Data/Templates/*.csv for the target schema).

Sources:
    Data/Options Data- {Apple,Tesla,SNP}.xlsx
        Sheet2: Date, Last Price            -> underlying spot
        Sheet4: Date, spot, then 12 Bloomberg ticker strings (one per
                Near/Far x ATM/OTM/ITM x Call/Put contract), e.g.
                "AAPL US 01/18/19 C160 Equity" -> strike/expiry/type
        Sheet5: Date, then 12 groups of 3 columns (Last, Bid, Ask) aligned
                1:1 with Sheet4's 12 contract columns. Confirmed against
                real values: col2 of each triple (Bid) is populated on
                essentially every date; col1 (Last) is usually
                '#N/A Invalid Security'; col3 (Ask) is only populated on
                the contract's roll/initiation day. We use Bid throughout
                for consistency.
    Data/Other Historical Data.xlsx
        Sheet1: paired date/value columns per series, incl. USGG10YR Index
                (10Y UST yield, %) used as the risk-free rate proxy.
    Data/Options_IV_data.xlsx
        "{AAPL,TSLA,SPX} Values" sheets: Date, then 44 Bloomberg BDP implied
                vol fields -- one per (moneyness in {80,90,95,100,105,110,120}%
                x DTE in {30D,60D,3M,6M,1Y,2Y}) = 42 columns, plus
                CALL_IMP_VOL_10D/PUT_IMP_VOL_10D (ATM-only, no moneyness
                wings at 10D). Column order confirmed against the legend row
                (row 15) of Sheet2 in the same workbook -- see IV_COLUMNS.
                The plain "{AAPL,TSLA,SPX}" sheets (without " Values") hold
                the same data via live Bloomberg formulas and have scattered
                cells corrupted into datetimes by Excel's autoformat (e.g.
                AAPL row 2637 cols 3/7) -- use the " Values" sheets, which
                are the pasted-static numbers, not those.
    Data/Yield_Curve_data.xlsx
        Sheet2: 10 side-by-side (Date, Yield) column pairs, one per tenor
                (EFFR, 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 10Y, 30Y -- see
                YIELD_CURVE_COLUMNS), each with its own independent date
                sequence (not row-aligned across tenors -- different series
                have gaps on different dates). Yields are quoted in percent.
                A handful of cells in the 5Y date column hold raw Excel date
                serials instead of datetimes (an autoformat glitch, same
                family of issue as the IV workbook above) -- see
                _coerce_excel_date().

IMPORTANT: option_prices.csv's ImpliedVol here is *back-derived from Bid via
Black-Scholes inversion* -- your raw data has no vendor IV field. This is a
placeholder until you pull real IV from Bloomberg; re-run once that column
exists (same output schema, just swap the source of ImpliedVol) rather than
inverting from a bid quote, which understates IV vs a mid/vendor value.

option_prices.csv also carries the Near/Far and ATM/OTM/ITM bucket each row
came from (Tenor, Moneyness columns), taken directly from CONTRACT_LABELS --
this is what DCC_GARCH.py's OPTION_BUCKET leg type keys off of.
"""

import datetime
import re

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from DCC_GARCH import bs_price

DATA_DIR = "Data"
OUT_DIR = "Data/Processed"

UNDERLYINGS = {
    "AAPL": f"{DATA_DIR}/Options Data- Apple.xlsx",
    "TSLA": f"{DATA_DIR}/Options Data- Tesla.xlsx",
    "SPX": f"{DATA_DIR}/Options Data- SNP.xlsx",
}

IV_WORKBOOK = f"{DATA_DIR}/Options_IV_data.xlsx"
IV_SHEETS = {"AAPL": "AAPL Values", "TSLA": "TSLA Values", "SPX": "SPX Values"}

# DTE label -> calendar days, matching the vol_surface.csv TenorDays convention.
TENOR_DAYS = {"10D": 10, "30D": 30, "60D": 60, "3M": 91, "6M": 182, "1Y": 365, "2Y": 730}

# (column index in the " Values" sheets) -> (DTE label, moneyness as %-of-spot),
# per the legend row (row 15 / index 14) of Sheet2 in Options_IV_data.xlsx. Column 0
# is Date. Columns 19/20 (10D) have no moneyness wings, only a Call/Put ATM quote --
# handled separately in build_vol_surface() since they average to one ATM node instead
# of mapping straight through like the other 42 columns.
IV_COLUMNS = {
    1: ("30D", 80), 2: ("60D", 80), 3: ("3M", 80), 4: ("6M", 80), 5: ("1Y", 80), 6: ("2Y", 80),
    7: ("30D", 90), 8: ("60D", 90), 9: ("3M", 90), 10: ("6M", 90), 11: ("1Y", 90), 12: ("2Y", 90),
    13: ("30D", 95), 14: ("60D", 95), 15: ("3M", 95), 16: ("6M", 95), 17: ("1Y", 95), 18: ("2Y", 95),
    21: ("30D", 100), 22: ("60D", 100), 23: ("3M", 100), 24: ("6M", 100), 25: ("1Y", 100), 26: ("2Y", 100),
    27: ("30D", 105), 28: ("60D", 105), 29: ("3M", 105), 30: ("6M", 105), 31: ("1Y", 105), 32: ("2Y", 105),
    33: ("30D", 110), 34: ("60D", 110), 35: ("3M", 110), 36: ("6M", 110), 37: ("1Y", 110), 38: ("2Y", 110),
    39: ("30D", 120), 40: ("60D", 120), 41: ("3M", 120), 42: ("6M", 120), 43: ("1Y", 120), 44: ("2Y", 120),
}
IV_CALL_10D_COL, IV_PUT_10D_COL = 19, 20

YIELD_CURVE_WORKBOOK = f"{DATA_DIR}/Yield_Curve_data.xlsx"
# (date column, yield column) letters in Sheet2, in the sheet's left-to-right order, and
# the tenor each pair belongs to (confirmed by matching the yield levels against the
# known historical UST curve shape on 2016-01-04, since the sheet's own header row only
# labels the 9 CMT columns -- 1M..30Y -- and doesn't label the leading EFFR column).
YIELD_CURVE_COLUMNS = [
    ("D", "E", "EFFR"), ("G", "H", "1M"), ("J", "K", "3M"), ("M", "N", "6M"),
    ("P", "Q", "1Y"), ("S", "T", "2Y"), ("V", "W", "3Y"), ("Y", "Z", "5Y"),
    ("AB", "AC", "10Y"), ("AE", "AF", "30Y"),
]
YIELD_TENOR_DAYS = {"EFFR": 1, "1M": 30, "3M": 91, "6M": 182, "1Y": 365, "2Y": 730, "3Y": 1095, "5Y": 1825, "10Y": 3650, "30Y": 10950}

TICKER_RE = re.compile(r"^(?P<root>\S+)\s+\S+\s+(?P<expiry>\d{2}/\d{2}/\d{2})\s+(?P<cp>[CP])(?P<strike>[\d.]+)\s+(?:Equity|Index)$")

CONTRACT_LABELS = [
    "Near ATM Call", "Near OTM Call", "Near ITM Call",
    "Near ATM Put", "Near ITM Put", "Near OTM Put",
    "Far ATM Call", "Far OTM Call", "Far ITM Call",
    "Far ATM Put", "Far ITM Put", "Far OTM Put",
]


def implied_vol(price, S, K, T, r, option_type):
    if T <= 0 or price is None or not np.isfinite(price) or price <= 0:
        return np.nan
    intrinsic = max(S - K, 0.0) if option_type == "C" else max(K - S, 0.0)
    if price <= intrinsic + 1e-6:
        return np.nan  # at/below intrinsic -- can't invert a sensible vol from a bid this low
    try:
        return brentq(lambda sigma: bs_price(S, K, T, r, sigma, option_type) - price, 1e-4, 5.0, xtol=1e-6)
    except ValueError:
        return np.nan


def extract_spot(path):
    df = pd.read_excel(path, sheet_name="Sheet2", header=0, usecols=[0, 1], names=["Date", "Spot"])
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")["Spot"].sort_index()


def extract_option_chain(path, underlying_ticker, spot_series, risk_free):
    tickers = pd.read_excel(path, sheet_name="Sheet4", header=None, skiprows=1)
    prices = pd.read_excel(path, sheet_name="Sheet5", header=None, skiprows=1)

    rows = []
    n = min(len(tickers), len(prices))
    for i in range(n):
        date = tickers.iat[i, 0]
        if pd.isna(date):
            continue
        date = pd.Timestamp(date)
        if date not in spot_series.index:
            continue
        spot = spot_series.loc[date]
        r = risk_free.get(date, np.nan)
        if not np.isfinite(spot) or not np.isfinite(r):
            continue

        for j in range(12):  # 12 contracts, per CONTRACT_LABELS order
            ticker_str = tickers.iat[i, 2 + j]
            if not isinstance(ticker_str, str):
                continue
            m = TICKER_RE.match(ticker_str.strip())
            if not m:
                continue

            tenor, moneyness, _cp_label = CONTRACT_LABELS[j].split()

            bid_col = 1 + 3 * j + 1  # [Last, Bid, Ask] per contract; prices sheet has no header col offset (already skiprows=1 aligned)
            bid = prices.iat[i, bid_col] if bid_col < prices.shape[1] else None
            if not isinstance(bid, (int, float)) or not np.isfinite(bid):
                continue

            expiry = pd.to_datetime(m.group("expiry"), format="%m/%d/%y")
            strike = float(m.group("strike"))
            cp = m.group("cp")
            T = (expiry - date).days / 365.0

            iv = implied_vol(bid, spot, strike, T, r, cp)
            if not np.isfinite(iv):
                continue

            rows.append({
                "Date": date,
                "OptionID": f"{underlying_ticker}_{expiry.date()}_{cp}{strike:g}",
                "UnderlyingTicker": underlying_ticker,
                "Type": cp,
                "Strike": strike,
                "Expiry": expiry,
                "Price": bid,
                "ImpliedVol": iv,
                "Tenor": tenor,
                "Moneyness": moneyness,
            })

    return pd.DataFrame(rows)


def build_vol_surface():
    """Options_IV_data.xlsx -> vol_surface.csv (Date, UnderlyingTicker, TenorDays,
    Moneyness, ImpliedVol). See IV_COLUMNS/IV_WORKBOOK docstring notes above."""
    frames = []
    for ticker, sheet in IV_SHEETS.items():
        df = pd.read_excel(IV_WORKBOOK, sheet_name=sheet, header=None)
        dates = pd.to_datetime(df[0])

        for col, (dte, mny_pct) in IV_COLUMNS.items():
            frames.append(pd.DataFrame({
                "Date": dates,
                "UnderlyingTicker": ticker,
                "TenorDays": TENOR_DAYS[dte],
                "Moneyness": mny_pct / 100.0 - 1.0,
                "ImpliedVol": df[col].to_numpy(dtype=float) / 100.0,
            }))

        # 10D has only an ATM call and put quote (no moneyness wings) -- average them
        # into a single ATM (Moneyness=0.0) node, consistent with put-call parity.
        atm_10d = (df[IV_CALL_10D_COL].to_numpy(dtype=float) + df[IV_PUT_10D_COL].to_numpy(dtype=float)) / 2.0
        frames.append(pd.DataFrame({
            "Date": dates,
            "UnderlyingTicker": ticker,
            "TenorDays": TENOR_DAYS["10D"],
            "Moneyness": 0.0,
            "ImpliedVol": atm_10d / 100.0,
        }))

    vol_surface = pd.concat(frames, ignore_index=True).dropna(subset=["ImpliedVol"])
    return vol_surface.sort_values(["Date", "UnderlyingTicker", "TenorDays", "Moneyness"])


def _coerce_excel_date(value):
    """A handful of cells in Yield_Curve_data.xlsx hold a raw Excel date serial (int)
    instead of a datetime -- an autoformat glitch. Excel's epoch is 1899-12-30 (the
    conventional off-by-one that also absorbs its fake 1900 leap day)."""
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.datetime(1899, 12, 30) + datetime.timedelta(days=value)
    return None


def build_risk_free_curve():
    """Yield_Curve_data.xlsx -> risk_free_rate.csv (Date, TenorDays, Rate). Each tenor's
    (Date, Yield) column pair has its own independent date sequence (see
    YIELD_CURVE_COLUMNS docstring note), so each series is forward/back-filled onto the
    union of all tenors' dates before being reshaped to long format -- otherwise most
    dates would only carry whichever handful of tenors happened to report that exact
    day, producing a ragged, day-to-day-inconsistent curve instead of a usable one."""
    import openpyxl

    wb = openpyxl.load_workbook(YIELD_CURVE_WORKBOOK, data_only=True)
    ws = wb["Sheet2"]

    series = {}
    for date_col, yield_col, tenor in YIELD_CURVE_COLUMNS:
        dates = [_coerce_excel_date(c.value) for c in ws[date_col]]
        yields = [c.value for c in ws[yield_col]]
        s = pd.Series(
            {pd.Timestamp(d): y for d, y in zip(dates, yields) if d is not None and y is not None}
        ).sort_index()
        series[tenor] = s[~s.index.duplicated(keep="last")]

    wide = pd.DataFrame(series).sort_index().ffill().bfill()

    long = wide.reset_index(names="Date").melt(id_vars="Date", var_name="Tenor", value_name="Rate")
    long["TenorDays"] = long["Tenor"].map(YIELD_TENOR_DAYS)
    long["Rate"] = long["Rate"] / 100.0
    return long[["Date", "TenorDays", "Rate"]].sort_values(["Date", "TenorDays"])


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    hist = pd.read_excel(f"{DATA_DIR}/Other Historical Data.xlsx", sheet_name="Sheet1", header=0, usecols=[0, 1], names=["Date", "USGG10YR"])
    hist = hist.dropna(subset=["Date"])
    hist["Date"] = pd.to_datetime(hist["Date"])
    risk_free = (hist.set_index("Date")["USGG10YR"] / 100.0).sort_index()
    risk_free = risk_free[~risk_free.index.duplicated(keep="last")]

    stock_rows = []
    chain_dfs = []
    for ticker, path in UNDERLYINGS.items():
        print(f"Processing {ticker} ({path})...")
        spot = extract_spot(path)
        spot = spot[~spot.index.duplicated(keep="last")]
        stock_rows.append(pd.DataFrame({"Date": spot.index, "Ticker": ticker, "AdjClose": spot.values}))

        rf_aligned = risk_free.reindex(spot.index).ffill().bfill()
        chain = extract_option_chain(path, ticker, spot, rf_aligned)
        print(f"  -> {len(chain)} option chain rows")
        chain_dfs.append(chain)

    stock_prices = pd.concat(stock_rows, ignore_index=True).sort_values(["Date", "Ticker"])
    stock_prices.to_csv(f"{OUT_DIR}/stock_prices.csv", index=False)

    option_prices = pd.concat(chain_dfs, ignore_index=True).sort_values(["Date", "UnderlyingTicker", "Type", "Strike"])
    option_prices.to_csv(f"{OUT_DIR}/option_prices.csv", index=False)

    print("Processing implied-vol surface (Options_IV_data.xlsx)...")
    vol_surface = build_vol_surface()
    vol_surface.to_csv(f"{OUT_DIR}/vol_surface.csv", index=False)

    print("Processing risk-free curve (Yield_Curve_data.xlsx)...")
    rf_curve = build_risk_free_curve()
    rf_curve.to_csv(f"{OUT_DIR}/risk_free_rate.csv", index=False)

    print(f"\nWrote {OUT_DIR}/stock_prices.csv ({len(stock_prices)} rows)")
    print(f"Wrote {OUT_DIR}/option_prices.csv ({len(option_prices)} rows)")
    print(f"Wrote {OUT_DIR}/vol_surface.csv ({len(vol_surface)} rows)")
    print(f"Wrote {OUT_DIR}/risk_free_rate.csv ({len(rf_curve)} rows)")


if __name__ == "__main__":
    main()
