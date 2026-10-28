"""
Stress tests: historical scenarios (cumulative moves of past crises, Equation 21) applied instantaneously to
tonight's positions, with proxies for assets that had no data at the time.
"""
import numpy as np
import pandas as pd

#__all__ = ["window_factor_moves"]
__all__ = ["window_factor_moves", "hypothetical_moves"]



def _move(series, start, end):
    """Return of a price series from the last close before ``start`` to the last close at or before ``end``."""
    s = series.dropna()
    before = s.loc[:pd.Timestamp(start) - pd.Timedelta(days=1)]
    upto = s.loc[:end]
    if before.empty or upto.empty:
        return np.nan
    return upto.iloc[-1] / before.iloc[-1] - 1


def window_factor_moves(prices, proxy_prices, proxies, eurusd, vix, start, end):
    """Cumulative factor moves over a historical window, in the factor format of the risk model.

    Parameters
    ----------
    prices : pd.DataFrame
        Local-currency prices of the assets (one column per asset key), full history.
    proxy_prices : pd.DataFrame
        Prices of the proxy tickers (one column per ticker).
    proxies : dict
        {asset: {'ticker', 'scale'}}: used when the asset has no price at the start of the window.
    eurusd : pd.Series
        US dollars per euro.
    vix : pd.Series
        VIX level.
    start, end : str
        First and last day of the window.

    Returns
    -------
    moves : pd.Series
        One value per factor: asset local returns, 'eurusd' (simple change), 'vix' (log change).
    used : dict
        {asset: proxy ticker} for the assets that were proxied.
    """
    moves, used = {}, {}
    for a in prices.columns:
        m = _move(prices[a], start, end)
        if np.isnan(m) and a in proxies:
            p = proxies[a]
            m = p["scale"] * _move(proxy_prices[p["ticker"]], start, end)
            used[a] = p["ticker"]
        moves[a] = m
    moves["eurusd"] = _move(eurusd, start, end)
    moves["vix"] = np.log(1 + _move(vix, start, end))
    return pd.Series(moves), used

def hypothetical_moves(shock, assets, sigma0):
    """Factor moves of a hypothetical scenario (Equation 22), in the factor format of the risk model.

    Parameters
    ----------
    shock : dict
        Any of: 'equity' (return of every Equity asset), 'rates_bp' (parallel rate shift, bp),
        'spreads_bp' (credit spread shift, bp), 'commodities' (return of the commodities asset),
        'eurusd' (change of USD per EUR), 'vix_level' (VIX after the shock, e.g. 0.45 for 45).
    assets : pd.DataFrame
        Asset table (asset_class, duration, convexity, spread_duration).
    sigma0 : float
        Tonight's VIX / 100.

    Returns
    -------
    pd.Series
        One value per factor: asset local returns, 'eurusd', 'vix' (log change).
    """
    a = assets.reindex(columns=["asset_class", "duration", "convexity", "spread_duration"])
    dur, cvx, sdur = (a[c].astype(float).fillna(0.0) for c in ["duration", "convexity", "spread_duration"])
    dy = shock.get("rates_bp", 0) / 1e4
    ds = shock.get("spreads_bp", 0) / 1e4
    moves = -dur * dy + 0.5 * cvx * dy ** 2 - sdur * ds                     # bonds (0 for the others)
    moves[a["asset_class"] == "Equity"] += shock.get("equity", 0.0)
    if "commodities" in moves.index:
        moves["commodities"] += shock.get("commodities", 0.0)
    moves["eurusd"] = shock.get("eurusd", 0.0)
    moves["vix"] = np.log(shock["vix_level"] / sigma0) if "vix_level" in shock else 0.0
    return moves