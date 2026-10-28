"""
One evening's risk: refit the models on the latest window, build tomorrow's FHS scenarios, revalue tonight's
positions (put included), and compute every measure of the report for that evening.
The same function run on two consecutive evenings gives the day-over-day change attribution (Equation 24).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .engine import _put_state
from .evt import evt_var_es
from .models import dcc_filter, fit_dcc, fit_gjr, gjr_variance
from .risk import pnl_contributions, scenario_pnl, var_es

__all__ = ["fit_models", "evening_risk", "var_of_positions"]


def fit_models(x_pct, prev=None):
    """GJR on every factor and DCC on their residuals, estimated on one window (Equations 5-8).

    Parameters
    ----------
    x_pct : pd.DataFrame
        Factor returns in % over the estimation window (the last row = tonight).
    prev : dict, optional
        A previous ``fit_models`` result, used as warm start.

    Returns
    -------
    dict
        'params' (DataFrame: mu, omega, alpha, gamma, beta, persistence, converged per factor),
        'dcc' (a, b, Qbar), 'sigma_next', 'R_next', 'Q_next' (tomorrow), 'e' (filtered shocks of the window),
        'converged' (all fits).
    """
    X = x_pct.to_numpy(dtype=float)
    cols = list(x_pct.columns)
    starts = [tuple(prev["params"].loc[c, ["mu", "omega", "alpha", "gamma", "beta"]]) for c in cols] if prev \
        else [None] * len(cols)
    fits = [fit_gjr(X[:, k], start=starts[k]) for k in range(len(cols))]
    par = pd.DataFrame([{k: f[k] for k in ["mu", "omega", "alpha", "gamma", "beta", "persistence", "converged"]}
                        for f in fits], index=cols)
    sig = np.column_stack([np.sqrt(gjr_variance(X[:, k], *par.loc[c, ["mu", "omega", "alpha", "gamma", "beta"]]))
                           for k, c in enumerate(cols)])
    z = (X - par["mu"].to_numpy()) / sig[:-1]
    d = fit_dcc(z, start=(prev["dcc"]["a"], prev["dcc"]["b"]) if prev else (0.02, 0.95))
    R, Q = dcc_filter(z, d["a"], d["b"], d["Qbar"], return_Q=True)
    e = np.linalg.solve(np.linalg.cholesky(R[:-1]), z[:, :, None])[:, :, 0]
    return {"params": par, "dcc": {"a": d["a"], "b": d["b"], "Qbar": d["Qbar"]},
            "sigma_next": sig[-1], "R_next": R[-1], "Q_next": Q[-1], "e": e,
            "converged": bool(par["converged"].all() and d["converged"])}


def var_of_positions(scen, currencies, positions, levels):
    """VaR / ES (FHS + EVT) of given positions on given scenarios: used for the change attribution."""
    L = -scenario_pnl(scen, currencies, positions["w"], positions["A"], positions["put"])["total"].to_numpy()
    t, _ = evt_var_es(L, levels)
    return t


def evening_risk(fac, currencies, hist, i, cfg, pm=None, put_cfg=None, models=None, prev=None):
    """Every risk number of the report for the evening of day ``i`` (forecasts for day i + 1).

    Parameters
    ----------
    fac : pd.DataFrame
        Factor returns (decimals), full history.
    currencies : pd.Series
        Currency of each asset.
    hist : dict
        ``fund_history`` output on the same calendar.
    i : int
        Position of the evening in ``fac`` (len(fac) - 1 = tonight).
    cfg : dict
        Configuration (models, measures).
    pm, put_cfg : optional
        Put market data and overlay configuration.
    models : dict, optional
        Models already fitted for this evening (``fit_models``); otherwise fitted here.
    prev : dict, optional
        Warm start for the fit.

    Returns
    -------
    dict
        'date', 'models', 'positions' (w, A, nav, put), 'scen' (FHS factor scenarios), 'by_pos', 'by_fac'
        (scenario P&L by position and by factor), 'table' (VaR / ES by method and level, EUR), 'nav'.
    """
    mc = cfg["models"]
    W = mc["fit_window"]
    dates = fac.index
    date = dates[i]
    next_date = dates[i + 1] if i + 1 < len(dates) else date + pd.offsets.BDay(1)
    if models is None:
        models = fit_models(100 * fac.iloc[i - W + 1:i + 1], prev)
    mu = models["params"]["mu"].to_numpy()
    L_next = np.linalg.cholesky(models["R_next"])
    scen = pd.DataFrame((mu + (models["e"] @ L_next.T) * models["sigma_next"]) / 100, columns=fac.columns)

    put = _put_state(hist, pm, put_cfg, i, next_date) if put_cfg is not None else None
    positions = {"w": hist["w_close"].iloc[i], "A": hist["assets"].iloc[i], "nav": hist["nav"].iloc[i], "put": put}
    by_pos, by_fac = pnl_contributions(scen, currencies, positions["w"], positions["A"], put)
    L_fhs = -by_pos.sum(axis=1).to_numpy()
    L_hs = -scenario_pnl(fac.iloc[i - mc["hs_window"] + 1:i + 1], currencies, positions["w"], positions["A"],
                         put)["total"].to_numpy()

    levels = cfg["measures"]["var_levels"]
    rows = {}
    for name, t_ in [("HS", var_es(L_hs, levels)), ("FHS", var_es(L_fhs, levels)),
                     ("FHS + EVT", evt_var_es(L_fhs, levels, mc["evt_tail_share"])[0])]:
        for q in levels:
            rows[(name, q)] = {"VaR": t_.loc[q, "VaR"], "ES": t_.loc[q, "ES"]}
    m_, s_ = L_fhs.mean(), L_fhs.std()
    for q in levels:
        zq = norm.ppf(q)
        rows[("Normal", q)] = {"VaR": m_ + s_ * zq, "ES": m_ + s_ * norm.pdf(zq) / (1 - q)}
    table = pd.DataFrame(rows).T
    table.index.names = ["method", "level"]
    return {"date": date, "models": models, "positions": positions, "scen": scen, "by_pos": by_pos,
            "by_fac": by_fac, "table": table, "nav": positions["nav"]}