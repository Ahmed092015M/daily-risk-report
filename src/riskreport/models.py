"""
Risk models: GJR-GARCH(1,1) for each factor by quasi-maximum likelihood (Equations 5-6)
and DCC(1,1) for their correlations (Equations 7-8). Returns are in % (e.g. -1.2 for -1.2%).
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.signal import lfilter

__all__ = ["gjr_variance", "gjr_negloglik", "fit_gjr", "fit_gjr_all", "dcc_filter", "dcc_negloglik", "fit_dcc", "ljung_box"]


# =========================================================================== #
# 1. GJR-GARCH(1,1)                                                           #
# =========================================================================== #
def gjr_variance(x, mu, omega, alpha, gamma, beta, s2_0=None):
    """Conditional variance of returns (Equation 5): the leverage term reacts to returns below the mean.

    Parameters
    ----------
    x : array
        Returns in %, in time order.
    mu, omega, alpha, gamma, beta : float
        Mean and GJR parameters.
    s2_0 : float, optional
        Variance of the first day. Default: backcast, an exponentially weighted average (weights 0.94^k)
        of the first min(75, n) squared residuals - the variance at the start of the sample (as in ``arch``).

    Returns
    -------
    np.ndarray
        sigma2_t for each day t, using data up to t - 1; length len(x) + 1, the last value being the
        forecast for the day after the sample.
    """
    e = np.asarray(x, dtype=float) - mu
    if s2_0 is None:
        tau = min(75, len(e))
        wts = 0.94 ** np.arange(tau)
        s2_0 = np.sum(wts / wts.sum() * e[:tau] ** 2)
    # s2[t] = u[t] + beta * s2[t-1] with u[t] = omega + (alpha + gamma 1[e_{t-1} < 0]) e_{t-1}^2:
    # a first-order linear filter, computed in compiled code by lfilter (same result as a Python loop)
    u = omega + (alpha + gamma * (e < 0)) * e ** 2
    y, _ = lfilter([1.0], [1.0, -beta], u, zi=[beta * s2_0])
    return np.concatenate([[s2_0], y])


def gjr_negloglik(params, x):
    """Negative normal (quasi) log-likelihood of GJR-GARCH(1,1) with a constant mean."""
    mu, omega, alpha, gamma, beta = params
    s2 = gjr_variance(x, mu, omega, alpha, gamma, beta)[:-1]
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return 1e10
    e = np.asarray(x, dtype=float) - mu
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + e ** 2 / s2)


def fit_gjr(x, start=None):
    """Quasi-maximum-likelihood fit of GJR-GARCH(1,1), robust: several starting points, best log-likelihood kept.

    With ``start`` (e.g. last week's parameters in a rolling refit) that run is tried first and returned
    directly if it converges; otherwise, and when no start is given, three standard starts are also tried.

    Parameters
    ----------
    x : array
        Returns in %.
    start : sequence, optional
        Starting values (mu, omega, alpha, gamma, beta).

    Returns
    -------
    dict
        'mu', 'omega', 'alpha', 'gamma', 'beta', 'persistence', 'loglik', 'converged', 'message',
        'n_starts' (runs tried), 'sigma' (in-sample sigma_t), 'sigma_next' (next-day forecast),
        'z' (standardised residuals).
    """
    x = np.asarray(x, dtype=float)
    m, v = x.mean(), x.var()
    starts = [(m, 0.05 * v, 0.03, 0.08, 0.88), (m, 0.02 * v, 0.05, 0.05, 0.90), (m, 0.10 * v, 0.10, 0.00, 0.80)]
    if start is not None:
        starts.insert(0, tuple(start))
    bounds = [(None, None), (1e-8, None), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)]
    cons = [{"type": "ineq", "fun": lambda p: 0.9999 - p[2] - p[3] / 2 - p[4]},     # stationarity
            {"type": "ineq", "fun": lambda p: p[2] + p[3]}]                          # alpha + gamma >= 0

    runs = []
    for k, s in enumerate(starts):
        res = optimize.minimize(gjr_negloglik, s, args=(x,), method="SLSQP", bounds=bounds,
                                constraints=cons, options={"maxiter": 1000, "ftol": 1e-10})
        if np.isfinite(res.fun):
            runs.append(res)
        if k == 0 and start is not None and res.success:              # warm start worked: done
            break
    best = min(runs, key=lambda r: r.fun)                              # highest log-likelihood
    ok_runs = [r.fun for r in runs if r.success]
    converged = bool(best.success or (ok_runs and best.fun <= min(ok_runs) + 1e-6))

    mu, omega, alpha, gamma, beta = best.x
    s2 = gjr_variance(x, mu, omega, alpha, gamma, beta)
    sigma = np.sqrt(s2)
    return {"mu": mu, "omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta,
            "persistence": alpha + gamma / 2 + beta, "loglik": -best.fun, "converged": converged,
            "message": best.message, "n_starts": len(runs),
            "sigma": sigma[:-1], "sigma_next": sigma[-1], "z": (x - mu) / sigma[:-1]}


def fit_gjr_all(returns_pct):
    """Fit GJR-GARCH to every column.

    Parameters
    ----------
    returns_pct : pd.DataFrame
        Daily returns in %, one column per factor.

    Returns
    -------
    params : pd.DataFrame
        One row per factor: mu, omega, alpha, gamma, beta, persistence, loglik, converged, message, sigma_next.
    z : pd.DataFrame
        Standardised residuals.
    sigma : pd.DataFrame
        In-sample conditional volatility (%).
    """
    rows, z, sig = {}, {}, {}
    for c in returns_pct.columns:
        f = fit_gjr(returns_pct[c].to_numpy())
        rows[c] = {k: f[k] for k in ["mu", "omega", "alpha", "gamma", "beta", "persistence",
                                     "loglik", "converged", "message", "sigma_next"]}
        z[c], sig[c] = f["z"], f["sigma"]
    idx = returns_pct.index
    return pd.DataFrame(rows).T.infer_objects(), pd.DataFrame(z, index=idx), pd.DataFrame(sig, index=idx)


# =========================================================================== #
# 2. DCC(1,1) correlations                                                    #
# =========================================================================== #
def dcc_filter(z, a, b, Qbar=None, return_Q=False):
    """DCC recursion (Equation 7): correlation matrices R_t from the standardised residuals.

    Parameters
    ----------
    z : array, shape (T, N)
        Standardised residuals.
    a, b : float
        DCC parameters (a >= 0, b >= 0, a + b < 1).
    Qbar : array (N, N), optional
        Unconditional matrix (default: z'z / T, correlation targeting).
    return_Q : bool
        Also return the Q_t matrices (needed to continue the recursion in a simulation).

    Returns
    -------
    R : np.ndarray, shape (T + 1, N, N)
        R_t for each day t using data up to t - 1; the last matrix is the forecast for the next day.
    Q : np.ndarray, shape (T + 1, N, N)
        Only if ``return_Q``.
    """
    z = np.asarray(z, dtype=float)
    T, N = z.shape
    if Qbar is None:
        Qbar = z.T @ z / T
    Q = np.empty((T + 1, N, N))
    Q[0] = Qbar
    for t in range(1, T + 1):
        Q[t] = (1 - a - b) * Qbar + a * np.outer(z[t - 1], z[t - 1]) + b * Q[t - 1]
    d = np.sqrt(np.einsum("tii->ti", Q))                          # diagonals of each Q_t
    R = Q / (d[:, :, None] * d[:, None, :])                        # R_t = diag(Q)^-1/2 Q diag(Q)^-1/2
    return (R, Q) if return_Q else R


def dcc_negloglik(params, z, Qbar):
    """Negative DCC (correlation part) quasi log-likelihood: 0.5 sum [ln|R_t| + z_t' R_t^-1 z_t - z_t' z_t]."""
    a, b = params
    if a < 0 or b < 0 or a + b >= 1:
        return 1e10
    R = dcc_filter(z, a, b, Qbar)[:-1]
    sign, logdet = np.linalg.slogdet(R)
    if np.any(sign <= 0):
        return 1e10
    Rinv_z = np.linalg.solve(R, z[:, :, None])[:, :, 0]
    return 0.5 * np.sum(logdet + np.einsum("ti,ti->t", z, Rinv_z) - np.einsum("ti,ti->t", z, z))


def fit_dcc(z, start=(0.02, 0.95)):
    """Two-step QML fit of DCC(1,1) with correlation targeting.

    Parameters
    ----------
    z : array or pd.DataFrame, shape (T, N)
        Standardised residuals from the univariate GJR fits.
    start : tuple
        Starting values (a, b).

    Returns
    -------
    dict
        'a', 'b', 'loglik' (correlation part), 'loglik_ccc' (constant correlation, a = b = 0), 'converged',
        'Qbar', 'R' (T x N x N, in-sample R_t), 'R_next' and 'Q_next' (next-day correlation and
        quasi-correlation matrices).
    """
    z = np.asarray(z, dtype=float)
    Qbar = z.T @ z / len(z)
    res = optimize.minimize(dcc_negloglik, start, args=(z, Qbar), method="SLSQP",
                            bounds=[(0.0, 1.0), (0.0, 1.0)],
                            constraints=[{"type": "ineq", "fun": lambda p: 0.9999 - p[0] - p[1]}],
                            options={"maxiter": 200, "ftol": 1e-10})
    a, b = res.x
    R, Q = dcc_filter(z, a, b, Qbar, return_Q=True)
    return {"a": a, "b": b, "loglik": -res.fun, "loglik_ccc": -dcc_negloglik((0.0, 0.0), z, Qbar),
            "converged": bool(res.success), "Qbar": Qbar, "R": R[:-1], "R_next": R[-1], "Q_next": Q[-1]}


# =========================================================================== #
# 3. Diagnostics                                                              #
# =========================================================================== #
def ljung_box(x, lags=20):
    """Ljung-Box test of no autocorrelation up to ``lags``: Q = n(n+2) sum rho_k^2 / (n-k) ~ chi2(lags).

    Returns
    -------
    (float, float)
        Q statistic and p-value.
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    denom = np.sum(x ** 2)
    rho = np.array([np.sum(x[k:] * x[:-k]) / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1)))
    return q, stats.chi2.sf(q, lags)