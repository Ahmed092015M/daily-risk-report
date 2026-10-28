"""
Market data: downloads from Yahoo Finance and FRED, and a summary of each series' history.
"""
import warnings

import numpy as np
import pandas as pd

#__all__ = ["load_yahoo", "load_fred", "history_summary"]
#__all__ = ["load_yahoo", "load_fred", "history_summary", "build_eurusd"]
#__all__ = ["load_yahoo", "load_fred", "history_summary", "build_eurusd", "eur_returns"]
#__all__ = ["load_yahoo", "load_fred", "history_summary", "build_eurusd", "eur_returns", "quality_checks"]
#__all__ = ["load_yahoo", "load_fred", "history_summary", "build_eurusd", "eur_returns", "mask_stale", "quality_checks"]
__all__ = ["load_yahoo", "load_fred", "history_summary", "build_eurusd", "eur_returns",
           "mask_stale", "quality_checks", "factor_returns", "eur_from_factors"]






def load_yahoo(tickers, start="2005-01-01"):
    """Download daily adjusted closes (total return: dividends reinvested) from Yahoo Finance.

    yfinance's own pandas deprecation warnings are silenced during the download only;
    failed tickers are still reported by yfinance and come back as empty (all-NaN) columns.

    Parameters
    ----------
    tickers : list of str
        Yahoo tickers, e.g. ["SPY", "EXW1.DE"].
    start : str
        First date requested (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Index = date (no time zone); one column per ticker; NaN before a ticker's first price.
    """
    import yfinance as yf                        # imported here so the package works without it

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # yfinance internals (inplace=... deprecations)
        df = yf.download(tickers, start=start, auto_adjust=False, progress=False, threads=False)["Adj Close"]
    if isinstance(df, pd.Series):                # a single ticker
        df = df.to_frame(tickers[0])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.reindex(columns=tickers).astype(float)


def load_fred(code, start="2005-01-01"):
    """Download one daily series from FRED (CSV endpoint, no API key).

    FRED writes holidays as "." - they become NaN and are dropped.

    Returns
    -------
    pd.Series
        Index = date; name = the FRED code.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    s = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
    s.index.name = "date"
    return s.loc[start:].dropna().astype(float).rename(code)


def history_summary(prices):
    """One row per series: first and last date, number of prices, longest gap, share of unchanged days.

    A series with no data (ticker not found) gets 0 prices and empty fields instead of an error.

    Parameters
    ----------
    prices : pd.DataFrame
        Prices, one column per series, NaN where missing.

    Returns
    -------
    pd.DataFrame
    """
    rows = {}
    for c in prices.columns:
        s = prices[c].dropna()
        if s.empty:
            rows[c] = {"first": None, "last": None, "prices": 0, "years": np.nan,
                       "longest gap (days)": np.nan, "unchanged days %": np.nan}
            continue
        gaps = s.index.to_series().diff().dt.days
        rows[c] = {"first": s.index[0].date(), "last": s.index[-1].date(), "prices": len(s),
                   "years": round((s.index[-1] - s.index[0]).days / 365.25, 1),
                   "longest gap (days)": int(gaps.max()) if len(s) > 1 else np.nan,
                   "unchanged days %": round(100 * (s.diff() == 0).mean(), 2)}
    return pd.DataFrame(rows).T

def build_eurusd(fred_fx, yahoo_fx, max_fill=5, tol=0.01):
    """EUR/USD from FRED (official, noon New York), extended with Yahoo for the days FRED has not published yet.

    Parameters
    ----------
    fred_fx : pd.Series
        FRED DEXUSEU, US dollars per euro.
    yahoo_fx : pd.Series
        Yahoo EURUSD=X, US dollars per euro (used only after FRED's last date).
    max_fill : int
        Maximum number of days filled from Yahoo; more means FRED has stopped updating.
    tol : float
        Maximum relative jump allowed between FRED's last value and Yahoo's first filled value.

    Returns
    -------
    eurusd : pd.Series
        The combined series, named "eurusd".
    provisional : pd.Series of bool
        True on the days taken from Yahoo.

    Raises
    ------
    ValueError
        If more than ``max_fill`` days need filling or the join jumps by more than ``tol``.
    """
    base = fred_fx.dropna()
    last = base.index[-1]
    fill = yahoo_fx.loc[yahoo_fx.index > last].dropna()
    if len(fill) > max_fill:
        raise ValueError(f"EUR/USD: {len(fill)} days after FRED's last date {last.date()} (max {max_fill}): "
                         f"check the FRED series")
    if len(fill):
        jump = fill.iloc[0] / base.iloc[-1] - 1
        if abs(jump) > tol:
            raise ValueError(f"EUR/USD: Yahoo {fill.iloc[0]:.4f} on {fill.index[0].date()} vs FRED "
                             f"{base.iloc[-1]:.4f} on {last.date()} ({jump:+.2%} > {tol:.0%}): suspected bad print")
    eurusd = pd.concat([base, fill]).rename("eurusd")
    provisional = pd.Series(eurusd.index > last, index=eurusd.index, name="provisional")
    return eurusd, provisional

def eur_returns(prices, eurusd, currencies):
    """Daily simple returns in EUR on the fund's calendar (the days when every asset has a price).

    USD prices are converted to EUR with that day's EUR/USD (US dollars per euro); the return of the
    EUR price is then Equation 1: (1 + r_USD) * X_{t-1} / X_t - 1.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted closes in local currency, one column per asset, NaN where a market is closed.
    eurusd : pd.Series
        US dollars per euro (from ``build_eurusd``).
    currencies : pd.Series
        Currency of each asset ("EUR" or "USD"), indexed like ``prices.columns``.

    Returns
    -------
    returns : pd.DataFrame
        Simple daily returns in EUR (the first common day is dropped: no return on it).
    info : dict
        'start', 'end', 'days', 'dropped' (days with at least one market closed), 'fx_filled'
        (days where EUR/USD was carried forward).
    """
    px = prices.dropna(how="any")                                  # days when every asset has a price
    dropped = int(prices.loc[px.index[0]:].notna().any(axis=1).sum()) - len(px)
    fx = eurusd.reindex(px.index)
    fx_filled = int(fx.isna().sum())
    fx = eurusd.reindex(px.index, method="ffill")                  # last known rate (no look-ahead)
    usd = currencies.index[currencies == "USD"]
    px_eur = px.copy()
    px_eur[usd] = px[usd].div(fx, axis=0)                          # USD price / (USD per EUR) = EUR price
    returns = px_eur.pct_change().iloc[1:]
    info = {"start": returns.index[0].date(), "end": returns.index[-1].date(), "days": len(returns),
            "dropped": dropped, "fx_filled": fx_filled}
    return returns, info

def mask_stale(prices, exchanges, min_assets=2):
    """Set prices to NaN on days when every asset of the same exchange shows an unchanged price.

    One ETF can close unchanged by chance; all the ETFs of one exchange unchanged on the same day
    means the data feed repeated the last price. Masking those days lets the fund calendar skip them,
    so the next return covers the whole period, exactly as for a holiday.

    Parameters
    ----------
    prices : pd.DataFrame
        Prices, one column per asset.
    exchanges : pd.Series
        Exchange of each asset (e.g. "Xetra", "US"), indexed like ``prices.columns``.
    min_assets : int
        Only exchanges with at least this many assets are tested.

    Returns
    -------
    clean : pd.DataFrame
        Prices with the stale days set to NaN for that exchange's assets.
    stale_days : dict of {exchange: list of dates}
    """
    unchanged = prices.diff().eq(0)
    clean = prices.copy()
    stale_days = {}
    for ex, cols in exchanges.groupby(exchanges).groups.items():
        cols = list(cols)
        if len(cols) < min_assets:
            continue
        days = unchanged[cols].all(axis=1) & prices[cols].notna().all(axis=1)
        clean.loc[days, cols] = np.nan
        stale_days[ex] = list(prices.index[days].date)
    return clean, stale_days


def quality_checks(returns, window=60, min_z=6.0, min_undo=0.8, max_zero_run=3, reviewed=None):
    """Suspect data errors and stale prices in a table of daily returns.

    A suspect error: |r_t| > min_z x the volatility of the previous ``window`` days, and the next day undoes
    more than ``min_undo`` of it in the opposite direction (one bad close creates a jump and its reversal).
    A stale period: ``max_zero_run`` or more zero returns in a row.
    Suspects are only reported, never removed: a person decides, and confirmed real moves are listed in
    the configuration (``reviewed``).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns, one column per asset.
    reviewed : list of dict, optional
        Entries {asset, date, note} from config["data_quality"]["reviewed"].

    Returns
    -------
    suspects : pd.DataFrame
        One row per suspect day: asset, date, return %, next day %, vol before %, z,
        status ("reviewed: real move" or "TO CHECK").
    stale : pd.DataFrame
        Per asset: longest run of zero returns and the date it ends; 'flag' True if >= max_zero_run.
    """
    ok = {(r["asset"], pd.Timestamp(r["date"]).date()) for r in (reviewed or [])}
    rows = []
    for c in returns.columns:
        r = returns[c]
        vol = r.rolling(window).std().shift(1)                     # volatility up to the day before
        nxt = r.shift(-1)
        z = r / vol
        hit = (z.abs() > min_z) & (np.sign(nxt) == -np.sign(r)) & (nxt.abs() > min_undo * r.abs())
        for d in r.index[hit.fillna(False)]:
            rows.append({"asset": c, "date": d.date(), "return %": 100 * r[d], "next day %": 100 * nxt[d],
                         "vol before %": 100 * vol[d], "z": z[d],
                         "status": "reviewed: real move" if (c, d.date()) in ok else "TO CHECK"})
    suspects = pd.DataFrame(rows, columns=["asset", "date", "return %", "next day %", "vol before %", "z", "status"])

    stale = {}
    for c in returns.columns:
        zero = returns[c] == 0
        run_id = (~zero).cumsum()
        runs = zero.groupby(run_id).sum()
        longest = int(runs.max())
        end = zero.groupby(run_id).apply(lambda s: s.index[-1]).loc[runs.idxmax()] if longest else None
        stale[c] = {"longest zero run": longest, "ends": end.date() if longest else None,
                    "flag": longest >= max_zero_run}
    return suspects, pd.DataFrame(stale).T

def factor_returns(prices, eurusd, vix):
    """Daily risk-factor returns on the fund's calendar (the days when every asset has a price).

    Factors: each asset's simple return in its own currency; EUR/USD simple change (US dollars per euro);
    VIX log change.

    Parameters
    ----------
    prices : pd.DataFrame
        Clean adjusted closes in local currency (NaN where a market is closed or a price is stale).
    eurusd : pd.Series
        US dollars per euro (``build_eurusd``).
    vix : pd.Series
        VIX level in points.

    Returns
    -------
    pd.DataFrame
        One column per asset, then 'eurusd' and 'vix'; the first common day is dropped.
    """
    px = prices.dropna(how="any")
    f = px.pct_change()
    f["eurusd"] = eurusd.reindex(px.index, method="ffill").pct_change()
    f["vix"] = np.log(vix.reindex(px.index, method="ffill")).diff()
    return f.iloc[1:]


def eur_from_factors(factors, currencies):
    """EUR returns of the assets from the factor returns (Equation 1): (1 + r_local) / (1 + r_EURUSD) - 1 for USD assets.

    Works on any table with the asset columns and 'eurusd' (historical days or simulated scenarios).

    Parameters
    ----------
    factors : pd.DataFrame
        Factor returns (assets in local currency, 'eurusd').
    currencies : pd.Series
        Currency of each asset.

    Returns
    -------
    pd.DataFrame
        EUR returns of the assets.
    """
    out = factors[list(currencies.index)].copy()
    usd = currencies.index[currencies == "USD"]
    out[usd] = (1 + factors[usd]).div(1 + factors["eurusd"], axis=0) - 1
    return out