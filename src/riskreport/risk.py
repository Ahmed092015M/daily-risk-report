"""
Tonight's risk: multivariate FHS scenarios (Equations 9-10), scenario P&L of the fund with the put revalued
in full (Equations 1, 14), and VaR / ES from scenario losses (Equation 4).
"""
import numpy as np
import pandas as pd

from .data import eur_from_factors
from .pricing import bs_put

#__all__ = ["fhs_scenarios", "scenario_pnl", "var_es"]
#__all__ = ["filtered_shocks", "fhs_scenarios", "simulate_paths", "scenario_pnl", "var_es"]
#__all__ = ["filtered_shocks", "fhs_scenarios", "simulate_paths", "compound_paths", "scenario_pnl", "var_es"]
__all__ = ["filtered_shocks", "fhs_scenarios", "simulate_paths", "compound_paths", "scenario_pnl", "var_es",
           "pnl_contributions", "euler_decomposition"]



def filtered_shocks(z, R):
    """Decorrelated standardised shocks e_s = L_s^-1 z_s (Equation 9): each past day's shock with its own
    volatility (already removed in z) and its own correlation removed.

    Parameters
    ----------
    z : array (T, N)
        Standardised residuals (GJR).
    R : array (T, N, N)
        DCC correlation matrix of each day.

    Returns
    -------
    np.ndarray (T, N)
        Shocks with (approximately) unit variance and no correlation.
    """
    L = np.linalg.cholesky(np.asarray(R, dtype=float))
    return np.linalg.solve(L, np.asarray(z, dtype=float)[:, :, None])[:, :, 0]


def fhs_scenarios(z, R, sigma_next, R_next, mu):
    """Filtered historical simulation (Equations 9-10): every past day's shock rescaled to tomorrow's
    volatility and correlation.

    Returns
    -------
    np.ndarray (T, N)
        Scenario returns for tomorrow, one row per past day (same units as mu, e.g. %).
    """
    e = filtered_shocks(z, R)
    L_next = np.linalg.cholesky(np.asarray(R_next, dtype=float))
    return np.asarray(mu) + (e @ L_next.T) * np.asarray(sigma_next)


def scenario_pnl(fac_scen, currencies, weights, assets_value, put=None):
    """P&L of the fund in EUR in each scenario: assets (EUR returns rebuilt with Equation 1) plus the put
    revalued in full (Equation 14).

    Parameters
    ----------
    fac_scen : pd.DataFrame
        Scenario factor returns in decimals: the asset columns (local currency), 'eurusd', 'vix' (log change).
    currencies : pd.Series
        Currency of each asset.
    weights : pd.Series
        Tonight's asset weights (share of the assets' value).
    assets_value : float
        Tonight's value of the assets in EUR.
    put : dict, optional
        Tonight's put: 'units', 'K', 'T' (years to expiry), 'dt' (years to the next valuation),
        'r', 'q', 'S0', 'sigma0', 'X0', 'value0' (EUR), 'index_factor' (factor driving the index, e.g. 'us_equity').

    Returns
    -------
    pd.DataFrame
        Columns 'assets', 'put', 'total' (P&L in EUR, one row per scenario).
    """
    r_eur = eur_from_factors(fac_scen, currencies)
    w = weights.reindex(r_eur.columns).to_numpy(dtype=float)
    pnl_assets = assets_value * (r_eur.to_numpy() @ w)
    pnl_put = np.zeros(len(fac_scen))
    if put is not None:
        S = put["S0"] * (1 + fac_scen[put["index_factor"]].to_numpy())
        sig = put["sigma0"] * np.exp(fac_scen["vix"].to_numpy())
        X = put["X0"] * (1 + fac_scen["eurusd"].to_numpy())
        value = put["units"] * bs_put(S, put["K"], max(put["T"] - put["dt"], 0.0), put["r"], put["q"], sig) / X
        pnl_put = value - put["value0"]
    return pd.DataFrame({"assets": pnl_assets, "put": pnl_put, "total": pnl_assets + pnl_put},
                        index=fac_scen.index)


def var_es(losses, levels):
    """VaR (empirical quantile) and ES (mean of the losses at or beyond VaR) of scenario losses (Equation 4).

    Parameters
    ----------
    losses : array
        Scenario losses (positive = loss).
    levels : sequence of float
        Confidence levels, e.g. [0.975, 0.99, 0.999].

    Returns
    -------
    pd.DataFrame
        Index = level; columns 'VaR', 'ES', 'n beyond' (number of scenarios at or beyond VaR).
    """
    losses = np.asarray(losses, dtype=float)
    rows = {}
    for q in levels:
        v = np.quantile(losses, q)
        tail = losses[losses >= v]
        rows[q] = {"VaR": v, "ES": tail.mean(), "n beyond": len(tail)}
    return pd.DataFrame(rows).T

def simulate_paths(e, mu, omega, alpha, gamma, beta, sigma_next, Q_next, Qbar, a, b,
                   horizon=20, n_paths=10_000, seed=42):
    """Multi-day paths of the factors (Equation 16): filtered bootstrap with GJR and DCC dynamics.

    Each day of each path: draw a past shock vector e_s at random, correlate it with the path's current R,
    scale it by the path's current sigma; then update sigma (GJR, Equation 5) and Q (DCC, Equation 7).

    Parameters
    ----------
    e : array (T, N)
        Filtered shocks (``filtered_shocks``).
    mu, omega, alpha, gamma, beta : arrays (N,)
        GJR parameters of each factor (returns in %).
    sigma_next : array (N,)
        Volatility forecast for the first simulated day.
    Q_next, Qbar : arrays (N, N)
        DCC quasi-correlation for the first day, and its unconditional level.
    a, b : float
        DCC parameters.
    horizon, n_paths, seed : int
        Days per path, number of paths, random seed (reproducible reports).

    Returns
    -------
    np.ndarray (n_paths, horizon, N)
        Simulated daily factor returns (same units as mu).
    """
    rng = np.random.default_rng(seed)
    e = np.asarray(e, dtype=float)
    T, N = e.shape
    mu, omega, alpha, gamma, beta = (np.asarray(v, dtype=float) for v in (mu, omega, alpha, gamma, beta))
    s2 = np.tile(np.asarray(sigma_next, dtype=float) ** 2, (n_paths, 1))
    Q = np.tile(np.asarray(Q_next, dtype=float), (n_paths, 1, 1))
    Qbar = np.asarray(Qbar, dtype=float)
    out = np.empty((n_paths, horizon, N))
    for h in range(horizon):
        d = np.sqrt(np.einsum("mii->mi", Q))
        L = np.linalg.cholesky(Q / (d[:, :, None] * d[:, None, :]))   # each path's R and its Cholesky
        z = np.einsum("mij,mj->mi", L, e[rng.integers(0, T, n_paths)])
        eps = np.sqrt(s2) * z
        out[:, h] = mu + eps
        s2 = omega + (alpha + gamma * (eps < 0)) * eps ** 2 + beta * s2
        Q = (1 - a - b) * Qbar + a * z[:, :, None] * z[:, None, :] + b * Q
    return out

def compound_paths(paths_pct, factor_names, log_factors=("vix",)):
    """Factor returns over the whole horizon from simulated daily paths.

    Simple returns are compounded, prod(1 + r) - 1; log changes (the VIX) are summed. The result has the same
    format as one-day scenarios, so ``scenario_pnl`` can be applied to it (Equation 1 is multiplicative,
    so it gives the compounded EUR return exactly).

    Parameters
    ----------
    paths_pct : array (n_paths, horizon, N)
        Daily factor returns in % (``simulate_paths``).
    factor_names : list of str
        Factor names, in column order.
    log_factors : tuple of str
        Factors expressed as log changes.

    Returns
    -------
    pd.DataFrame
        One row per path: horizon return of each factor (decimals).
    """
    P = np.asarray(paths_pct, dtype=float) / 100
    out = pd.DataFrame(np.prod(1 + P, axis=1) - 1, columns=factor_names)
    for f in log_factors:
        if f in out:
            out[f] = P[:, :, factor_names.index(f)].sum(axis=1)
    return out

def pnl_contributions(fac_scen, currencies, weights, assets_value, put=None):
    """P&L of every position in every scenario, and the split of each USD asset into local-currency and FX parts.

    Parameters
    ----------
    fac_scen, currencies, weights, assets_value, put
        As in ``scenario_pnl``.

    Returns
    -------
    by_position : pd.DataFrame
        One column per asset (EUR P&L, FX included) and 'put'; the row sums equal ``scenario_pnl(...)['total']``.
    by_factor : pd.DataFrame
        One column per asset (P&L from its local-currency return), 'fx' (the FX part of all USD assets) and 'put'.
    """
    cols = list(currencies.index)
    exposure = assets_value * weights.reindex(cols).to_numpy(dtype=float)        # EUR invested in each asset
    r_eur = eur_from_factors(fac_scen, currencies)
    pos = r_eur[cols] * exposure
    local = fac_scen[cols] * exposure
    put_pnl = scenario_pnl(fac_scen, currencies, weights, assets_value, put)["put"] if put is not None \
        else pd.Series(0.0, index=fac_scen.index)
    by_position = pos.assign(put=put_pnl.to_numpy())
    by_factor = local.assign(fx=(pos - local).sum(axis=1), put=put_pnl.to_numpy())
    return by_position, by_factor


def euler_decomposition(contrib, var_level=0.99, es_level=0.975, band=0.01):
    """Component VaR / ES and incremental VaR of each column of a scenario P&L table (Equations 17, 19, 20).

    Parameters
    ----------
    contrib : pd.DataFrame
        P&L of each component in each scenario (EUR); the row sums are the fund's P&L.
    var_level, es_level : float
        Levels of the VaR and ES decomposed.
    band : float
        Share of scenarios around the VaR quantile used for component VaR (0.01 = the 1% of scenarios
        whose loss is closest to VaR).

    Returns
    -------
    pd.DataFrame
        One row per component: 'component VaR', 'share of VaR', 'component ES', 'share of ES', 'incremental VaR';
        the totals are in ``.attrs`` ('VaR', 'ES').
    """
    C = contrib.to_numpy(dtype=float)
    L = -C.sum(axis=1)
    var = np.quantile(L, var_level)
    var_es_level = np.quantile(L, es_level)
    tail = L >= var_es_level
    es = L[tail].mean()
    ces = -C[tail].mean(axis=0)                                      # sums exactly to ES

    k = max(int(band * len(L)), 1)
    near = np.argsort(np.abs(L - var))[:k]                           # scenarios whose loss is closest to VaR
    raw = -C[near].mean(axis=0)
    cvar = raw / raw.sum() * var                                     # scaled to add up to VaR

    ivar = np.array([var - np.quantile(L + C[:, j], var_level) for j in range(C.shape[1])])   # VaR with - without
    out = pd.DataFrame({"component VaR": cvar, "share of VaR": cvar / var, "component ES": ces,
                        "share of ES": ces / es, "incremental VaR": ivar}, index=contrib.columns)
    out.attrs.update({"VaR": var, "ES": es})
    return out