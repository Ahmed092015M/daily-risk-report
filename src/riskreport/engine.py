"""
Out-of-sample engine: every day's VaR / ES forecast of the fund, using only data up to the day before
(no look-ahead), for four methods - HS, normal, FHS (GJR + DCC) and FHS + EVT - with the put revalued in full.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .evt import evt_var_es
from .models import dcc_filter, fit_dcc, fit_gjr, gjr_variance
from .risk import scenario_pnl, var_es

__all__ = ["METHODS", "rolling_risk"]

METHODS = ["HS", "Normal", "FHS", "FHS + EVT"]


def _put_state(hist, pm, put_cfg, i, next_date):
    """The put held at the close of day i (positions for day i + 1), with that close's market data."""
    d = hist["nav"].index[i]
    mk = pm.loc[:d].iloc[-1]
    expiry = hist["put_expiry"].iloc[i]
    return {"units": hist["put_units"].iloc[i], "K": hist["put_strike"].iloc[i],
            "T": (expiry - d).days / 365, "dt": (next_date - d).days / 365,
            "r": mk["r"], "q": put_cfg["dividend_yield"], "S0": mk["S"], "sigma0": mk["sigma"], "X0": mk["X"],
            "value0": hist["put_value"].iloc[i], "index_factor": put_cfg["underlying"]}


def rolling_risk(fac, currencies, hist, pm=None, put_cfg=None, window=1000, refit_every=20, hs_window=500,
                 levels=(0.975, 0.99, 0.999), pool="rolling", evt_share=0.10, start=None, end=None):
    """Daily out-of-sample VaR and ES of the fund, and the realised loss of the next day.

    Every ``refit_every`` days: GJR on each factor and DCC, estimated on the previous ``window`` days (warm starts).
    Every day: volatility and correlation updated with yesterday's data; FHS scenarios from the filtered shocks
    (``pool``: "rolling" = the estimation window, "expanding" = every filtered shock since the first window);
    the positions held at the previous close (weights, assets, put) revalued in each scenario.

    Parameters
    ----------
    fac : pd.DataFrame
        Daily factor returns in decimals (assets in local currency, 'eurusd', 'vix' log change).
    currencies : pd.Series
        Currency of each asset.
    hist : dict
        Output of ``fund_history`` on the same calendar (positions held at each close, NAV).
    pm, put_cfg : optional
        Put market data (``put_market``) and the overlay configuration; None = no put.
    window, refit_every, hs_window : int
        Estimation window, days between re-estimations, HS window.
    levels : tuple of float
        VaR / ES levels.
    pool : str
        "rolling" or "expanding" pool of filtered shocks.
    evt_share : float
        Share of scenario losses in the GPD tail.
    start, end : int, optional
        First and last+1 positions forecast (default: window, len(fac)).

    Returns
    -------
    out : pd.DataFrame
        Index = forecast date; column ('loss', '', '') = realised loss of the fund in EUR (positive = loss),
        and (method, 'VaR' / 'ES', level) = the forecasts made the evening before.
    params : pd.DataFrame
        DCC a, b, the GJR persistence of each factor and a convergence flag at each re-estimation.
    """
    X = 100 * fac.to_numpy(dtype=float)
    dates = fac.index
    nav = hist["nav"].to_numpy()
    start = window if start is None else start
    end = len(fac) if end is None else end
    levels = list(levels)
    n_f = X.shape[1]

    rows, plog = [], []
    prev, prev_dcc, e_hist = [None] * n_f, (0.02, 0.95), []
    for j in range(start, end, refit_every):
        b_end = min(j + refit_every, end)
        seg = X[j - window:b_end]                                       # estimation window + forecast block
        fits = [fit_gjr(seg[:window, k], start=prev[k]) for k in range(n_f)]
        prev = [(f["mu"], f["omega"], f["alpha"], f["gamma"], f["beta"]) for f in fits]
        mu = np.array([p[0] for p in prev])
        sig = np.column_stack([np.sqrt(gjr_variance(seg[:, k], *prev[k])) for k in range(n_f)])   # len(seg) + 1
        z = (seg - mu) / sig[:-1]
        d = fit_dcc(z[:window], start=prev_dcc)
        prev_dcc = (d["a"], d["b"])
        R = dcc_filter(z, d["a"], d["b"], d["Qbar"])                    # R[i]: day i of seg, data up to i - 1
        L_all = np.linalg.cholesky(R)
        e_seg = np.linalg.solve(L_all[:-1], z[:, :, None])[:, :, 0]    # filtered shock of every day of seg
        if pool == "expanding" and not e_hist:
            e_hist.extend(e_seg[:window])
        plog.append({"date": dates[j], "a": d["a"], "b": d["b"],
                     **{f"pers {c}": f["persistence"] for c, f in zip(fac.columns, fits)},
                     "all converged": all(f["converged"] for f in fits) and d["converged"]})

        for t in range(j, b_end):
            i = t - (j - window)                                         # position of day t in seg
            shocks = e_seg[:window] if pool == "rolling" else np.asarray(e_hist)
            scen = pd.DataFrame((mu + (shocks @ L_all[i].T) * sig[i]) / 100, columns=fac.columns)
            w, A = hist["w_close"].iloc[t - 1], hist["assets"].iloc[t - 1]
            put = _put_state(hist, pm, put_cfg, t - 1, dates[t]) if put_cfg is not None else None
            L_fhs = -scenario_pnl(scen, currencies, w, A, put)["total"].to_numpy()
            L_hs = -scenario_pnl(fac.iloc[t - hs_window:t], currencies, w, A, put)["total"].to_numpy()

            row = {("loss", "", ""): -(nav[t] - nav[t - 1])}
            for name, L in [("HS", L_hs), ("FHS", L_fhs)]:
                t_ = var_es(L, levels)
                for q in levels:
                    row[(name, "VaR", q)], row[(name, "ES", q)] = t_.loc[q, "VaR"], t_.loc[q, "ES"]
            m_, s_ = L_fhs.mean(), L_fhs.std()
            for q in levels:
                zq = norm.ppf(q)
                row[("Normal", "VaR", q)] = m_ + s_ * zq
                row[("Normal", "ES", q)] = m_ + s_ * norm.pdf(zq) / (1 - q)
            t_evt, _ = evt_var_es(L_fhs, levels, evt_share)
            for q in levels:
                row[("FHS + EVT", "VaR", q)], row[("FHS + EVT", "ES", q)] = t_evt.loc[q, "VaR"], t_evt.loc[q, "ES"]
            rows.append(row)
            if pool == "expanding":
                e_hist.append(e_seg[i])                                  # day t's shock, known after its close

    out = pd.DataFrame(rows, index=dates[start:end])
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["method", "measure", "level"])
    return out, pd.DataFrame(plog).set_index("date")