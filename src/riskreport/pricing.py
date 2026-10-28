"""
Black-Scholes pricing of the protective put (Equation 13), vectorised for scenario revaluation.
"""
import numpy as np
from scipy.stats import norm

__all__ = ["bs_put", "bs_put_delta"]


def _prepare(S, K, T, r, q, sigma):
    """Inputs as float arrays of at least one dimension, broadcast together; and whether all were scalars."""
    args = (S, K, T, r, q, sigma)
    scalar = all(np.ndim(a) == 0 for a in args)
    arrays = np.broadcast_arrays(*(np.atleast_1d(np.asarray(a, dtype=float)) for a in args))
    return scalar, arrays


def _d1_d2(S, K, T, r, q, sigma):
    """d1 and d2 of Black-Scholes, for T > 0 and sigma > 0."""
    vol_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol_t
    return d1, d1 - vol_t


def bs_put(S, K, T, r, q, sigma):
    """Black-Scholes value of a European put with a continuous dividend yield.

    All inputs can be scalars or arrays (broadcast together). Where T <= 0 the payoff max(K - S, 0) is returned.

    Parameters
    ----------
    S : float or array
        Index level.
    K : float or array
        Strike.
    T : float or array
        Time to expiry in years.
    r : float or array
        Continuously compounded risk-free rate (e.g. ln(1 + 4.22 / 100)).
    q : float or array
        Continuous dividend yield.
    sigma : float or array
        Volatility (e.g. VIX / 100).

    Returns
    -------
    float or np.ndarray
        Put value, in index points (a float if every input was a scalar).
    """
    scalar, (S, K, T, r, q, sigma) = _prepare(S, K, T, r, q, sigma)
    out = np.maximum(K - S, 0.0)                                   # payoff at expiry (a new array)
    live = T > 0
    if live.any():
        d1, d2 = _d1_d2(S[live], K[live], T[live], r[live], q[live], sigma[live])
        out[live] = (K[live] * np.exp(-r[live] * T[live]) * norm.cdf(-d2)
                     - S[live] * np.exp(-q[live] * T[live]) * norm.cdf(-d1))
    return float(out[0]) if scalar else out


def bs_put_delta(S, K, T, r, q, sigma):
    """Delta of the put, dP/dS = -exp(-qT) N(-d1); at expiry -1 if in the money, else 0.

    Inputs as in ``bs_put``.

    Returns
    -------
    float or np.ndarray
        Put delta (between -1 and 0).
    """
    scalar, (S, K, T, r, q, sigma) = _prepare(S, K, T, r, q, sigma)
    out = -(S < K).astype(float)                                   # at expiry: -1 in the money, 0 otherwise
    live = T > 0
    if live.any():
        d1, _ = _d1_d2(S[live], K[live], T[live], r[live], q[live], sigma[live])
        out[live] = -np.exp(-q[live] * T[live]) * norm.cdf(-d1)
    return float(out[0]) if scalar else out