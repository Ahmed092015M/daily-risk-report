"""
Extreme value theory on scenario losses: GPD fit by maximum likelihood and GPD VaR / ES (Equation 12).
Same estimator as Project 7 (validated there against scipy).
"""
import numpy as np
import pandas as pd
from scipy import optimize

__all__ = ["fit_gpd", "gpd_var_es", "evt_var_es"]


def _gpd_negloglik(params, y):
    """Negative GPD log-likelihood of the excesses y; +inf outside the valid region."""
    xi, beta = params
    if beta <= 0:
        return np.inf
    z = 1 + xi * y / beta
    if np.any(z <= 0):
        return np.inf
    if abs(xi) < 1e-8:                                          # exponential limit
        return len(y) * np.log(beta) + y.sum() / beta
    return len(y) * np.log(beta) + (1 + 1 / xi) * np.log(z).sum()


def fit_gpd(y):
    """Maximum-likelihood GPD fit to excesses y > 0 (Nelder-Mead from method-of-moments starting values).

    Returns
    -------
    dict
        'xi', 'beta', 'n', 'converged'.
    """
    y = np.asarray(y, dtype=float)
    m, v = y.mean(), y.var()
    start = [0.5 * (1 - m ** 2 / v), 0.5 * m * (1 + m ** 2 / v)]
    with np.errstate(invalid="ignore"):
        res = optimize.minimize(_gpd_negloglik, start, args=(y,), method="Nelder-Mead",
                                options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    return {"xi": res.x[0], "beta": res.x[1], "n": len(y), "converged": bool(res.success)}


def gpd_var_es(u, xi, beta, n, n_u, q):
    """GPD VaR and ES (Equation 12): VaR = u + beta/xi [((n/n_u)(1-q))^-xi - 1]; ES = (VaR + beta - xi u)/(1 - xi)."""
    q = np.asarray(q, dtype=float)
    ratio = (n / n_u) * (1 - q)
    if abs(xi) < 1e-8:
        var = u - beta * np.log(ratio)
        return var, var + beta
    var = u + beta / xi * (ratio ** (-xi) - 1)
    es = (var + beta - xi * u) / (1 - xi) if xi < 1 else np.full_like(var, np.inf)
    return var, es


def evt_var_es(losses, levels, tail_share=0.10):
    """VaR and ES from a GPD fitted to the worst ``tail_share`` of scenario losses.

    Returns
    -------
    table : pd.DataFrame
        Index = level; columns 'VaR', 'ES'.
    fit : dict
        GPD fit with 'u' (threshold) and 'n_total' added.
    """
    L = np.asarray(losses, dtype=float)
    u = np.quantile(L, 1 - tail_share)
    y = L[L > u] - u
    f = fit_gpd(y)
    var, es = gpd_var_es(u, f["xi"], f["beta"], len(L), len(y), levels)
    f.update({"u": u, "n_total": len(L)})
    return pd.DataFrame({"VaR": np.atleast_1d(var), "ES": np.atleast_1d(es)}, index=list(levels)), f