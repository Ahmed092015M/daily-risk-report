"""
The fund: target weights, daily drift, monthly rebalancing and NAV (Equations 2-3),
with an optional protective put on one sleeve, rolled at each month-end (Equations 13-14).
"""
import numpy as np
import pandas as pd

from .pricing import bs_put

__all__ = ["put_market", "fund_history"]


def put_market(index, vix, rate_pct, eurusd):
    """Market inputs of the put on one daily calendar (forward-filled: last known value, no look-ahead).

    Parameters
    ----------
    index : pd.Series
        S&P 500 level.
    vix : pd.Series
        VIX in points (e.g. 17.58).
    rate_pct : pd.Series
        3-month T-bill in % (e.g. 4.22).
    eurusd : pd.Series
        US dollars per euro.

    Returns
    -------
    pd.DataFrame
        Columns S, sigma (= VIX / 100), r (continuously compounded, ln(1 + y / 100)), X (USD per EUR).
    """
    m = pd.concat({"S": index, "sigma": vix / 100, "r": np.log(1 + rate_pct / 100), "X": eurusd},
                  axis=1, sort=True)
    return m.ffill().dropna()


def fund_history(returns, target, nav0=100_000_000, put=None):
    """Day-by-day history of the fund with weights drifting between month-end rebalancings,
    and optionally a protective put rolled at each month-end.

    Convention: the asset weights at the close of day t are used for the return of day t + 1 (Equation 2);
    they drift with the assets' returns (Equation 3) and are reset to the target at the close of the last
    fund day of each month. With a put, NAV = value of the assets + value of the put (in EUR); at each
    month-end the old put is sold, a new one bought, and the difference is paid from (or into) the assets.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns in EUR, one column per asset.
    target : pd.Series
        Target weights of the assets, indexed like ``returns.columns``, summing to 1.
    nav0 : float
        NAV at the close of the day before the first return.
    put : dict, optional
        'market' (output of ``put_market``), 'underlying' (asset key of the protected sleeve),
        'moneyness', 'maturity_months', 'hedge_ratio', 'dividend_yield'. None = no overlay.

    Returns
    -------
    hist : dict
        'nav', 'ret' (fund daily return), 'assets' (value of the assets), 'w_close' (asset weights at each
        close after rebalancing), 'w_drift' (before rebalancing), 'turnover';
        with a put also 'put_value' (EUR, after any roll), 'put_pnl' (EUR, the day's change before the roll),
        'put_units' (index units hedged), 'put_strike', 'put_expiry'.
    """
    r = returns.to_numpy()
    tgt = target.reindex(returns.columns).to_numpy(dtype=float)
    idx, cols = returns.index, returns.columns
    n = len(returns)
    month = idx.to_period("M")
    next_bday = idx[-1] + pd.offsets.BDay(1)                       # the last day is a month-end only if the
    month_end = np.r_[month[1:] != month[:-1],                     # next business day is in a new month
                      next_bday.to_period("M") != month[-1]]

    if put is not None:
        mk = put["market"]
        start = mk.index[mk.index < idx[0]][-1]                        # the close before the first return
        mk = mk.reindex(idx.insert(0, start), method="ffill")
        u = list(cols).index(put["underlying"])
        q, mny, h = put["dividend_yield"], put["moneyness"], put["hedge_ratio"]
        months = put["maturity_months"]

        def buy(date, total):
            """Units, strike, expiry and EUR cost of a new put protecting h x target sleeve of ``total``."""
            S, sig, rr, X = mk.loc[date, ["S", "sigma", "r", "X"]]
            units = h * tgt[u] * total * X / S                         # EUR -> USD -> index units
            K, expiry = mny * S, date + pd.DateOffset(months=months)
            cost = units * bs_put(S, K, (expiry - date).days / 365, rr, q, sig) / X
            return units, K, expiry, cost

        units, K, expiry, Vp = buy(start, nav0)
        A = nav0 - Vp                                                  # premium paid from the fund
    else:
        A, Vp = nav0, 0.0

    w = tgt.copy()
    out = {k: np.empty(n) for k in ["nav", "ret", "assets", "turnover", "put_value", "put_pnl",
                                    "put_units", "put_strike"]}
    expiries = []
    w_drift = np.empty_like(r); w_close = np.empty_like(r)
    nav_prev = A + Vp
    for t in range(n):
        d = idx[t]
        rp = w @ r[t]                                                  # Equation 2 (assets only)
        A *= 1 + rp
        w = w * (1 + r[t]) / (1 + rp)                                  # Equation 3
        w_drift[t] = w
        pnl = 0.0
        if put is not None:
            S, sig, rr, X = mk.loc[d, ["S", "sigma", "r", "X"]]
            V_new = units * bs_put(S, K, (expiry - d).days / 365, rr, q, sig) / X   # Equation 14
            pnl, Vp = V_new - Vp, V_new
        nav = A + Vp
        out["turnover"][t] = 0.0
        if month_end[t]:
            out["turnover"][t] = 0.5 * np.abs(w - tgt).sum()          # one-way turnover of the assets
            if put is not None:                                        # roll: sell the old put, buy a new one
                A += Vp
                units, K, expiry, Vp = buy(d, A)
                A -= Vp
            w = tgt.copy()
        w_close[t] = w
        out["nav"][t] = nav
        out["ret"][t] = nav / nav_prev - 1
        out["assets"][t] = A
        out["put_value"][t], out["put_pnl"][t] = Vp, pnl
        out["put_units"][t] = units if put is not None else 0.0
        out["put_strike"][t] = K if put is not None else np.nan
        expiries.append(expiry if put is not None else pd.NaT)
        nav_prev = nav

    hist = {"nav": pd.Series(out["nav"], idx, name="nav"), "ret": pd.Series(out["ret"], idx, name="fund"),
            "assets": pd.Series(out["assets"], idx, name="assets"),
            "w_close": pd.DataFrame(w_close, idx, cols), "w_drift": pd.DataFrame(w_drift, idx, cols),
            "turnover": pd.Series(out["turnover"], idx, name="turnover")}
    if put is not None:
        hist.update({"put_value": pd.Series(out["put_value"], idx, name="put_value"),
                     "put_pnl": pd.Series(out["put_pnl"], idx, name="put_pnl"),
                     "put_units": pd.Series(out["put_units"], idx, name="put_units"),
                     "put_strike": pd.Series(out["put_strike"], idx, name="put_strike"),
                     "put_expiry": pd.Series(expiries, idx, name="put_expiry")})
    return hist