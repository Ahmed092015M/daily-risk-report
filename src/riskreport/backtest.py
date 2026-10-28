"""
VaR and ES backtests for losses (positive = loss): Kupiec, Christoffersen, exact binomial, Acerbi-Szekely Z2
(Equations 27-30, as in Project 7) and the Basel traffic light (Equation 25).
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

__all__ = ["kupiec_test", "christoffersen_test", "binomial_pvalue", "acerbi_szekely_z2", "z2_critical",
           "backtest_table", "traffic_light"]


# =========================================================================== #
# 1. VaR tests                                                                #
# =========================================================================== #
def _lr_uc(h, p):
    """Kupiec likelihood ratio: binomial likelihood at p vs at the observed breach rate."""
    n, x = len(h), int(h.sum())
    return -2 * ((xlogy(n - x, 1 - p) + xlogy(x, p)) - (xlogy(n - x, 1 - x / n) + xlogy(x, x / n)))


def kupiec_test(hits, p):
    """Kupiec unconditional coverage test (Equation 27): (breaches, p-value), LR ~ chi2(1)."""
    h = np.asarray(hits, dtype=int)
    return int(h.sum()), float(stats.chi2.sf(_lr_uc(h, p), 1))


def christoffersen_test(hits, p):
    """Christoffersen conditional coverage test (Equation 28): (consecutive breaches, p-value), LR ~ chi2(2)."""
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum()); n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum()); n11 = int(((prev == 1) & (curr == 1)).sum())
    pi01, pi11 = n01 / max(n00 + n01, 1), n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    lr_ind = -2 * ((xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi))
                   - (xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)))
    return n11, float(stats.chi2.sf(_lr_uc(h, p) + lr_ind, 2))


def binomial_pvalue(hits, p):
    """Exact two-sided binomial p-value of the breach count (Equation 30)."""
    h = np.asarray(hits, dtype=int)
    n, x = len(h), int(h.sum())
    return float(min(1.0, 2 * min(stats.binom.cdf(x, n, p), stats.binom.sf(x - 1, n, p))))


# =========================================================================== #
# 2. ES test                                                                  #
# =========================================================================== #
def acerbi_szekely_z2(loss, var, es, p):
    """Acerbi-Szekely Z2 for ES written for losses (Equation 29): 0 if ES is right, negative if ES is too low."""
    x, v, e = (np.asarray(a, dtype=float) for a in (loss, var, es))
    return float(1 - (x * (x > v) / (p * e)).sum() / len(x))


def z2_critical(T, p, n_sim=2000, alpha=0.05, seed=0):
    """5% critical value of Z2 by simulation under a correct model (normal losses, correct VaR / ES).

    Returns
    -------
    (float, float)
        Critical value (alpha quantile) and the mean of the simulated Z2.
    """
    rng = np.random.default_rng(seed)
    z = stats.norm.ppf(1 - p)
    var, es = z, stats.norm.pdf(z) / p
    sims = np.array([acerbi_szekely_z2(rng.standard_normal(T), var, es, p) for _ in range(n_sim)])
    return float(np.quantile(sims, alpha)), float(sims.mean())


# =========================================================================== #
# 3. Summary and traffic light                                                #
# =========================================================================== #
def backtest_table(loss, var_es, methods, levels, z2_crit):
    """Backtests of every method at every level: breaches, Kupiec, binomial, Christoffersen, Z2, verdict.

    Parameters
    ----------
    loss : pd.Series
        Realised losses on the forecast days.
    var_es : pd.DataFrame
        Forecasts with columns (method, 'VaR' / 'ES', level).
    methods, levels : sequence
        Methods and levels tested.
    z2_crit : dict of {level: float}
        5% critical values of Z2.

    Returns
    -------
    pd.DataFrame
        Index = (level, method); one column per statistic and 'pass all'.
    """
    rows = []
    for q in levels:
        p = 1 - q
        for m in methods:
            var, es = var_es[(m, "VaR", q)], var_es[(m, "ES", q)]
            hits = (loss > var).astype(int).values
            x_b, p_k = kupiec_test(hits, p)
            n11, p_c = christoffersen_test(hits, p)
            z2 = acerbi_szekely_z2(loss, var, es, p)
            rows.append({"level": q, "method": m, "breaches": x_b, "expected": p * len(loss),
                         "ratio": x_b / (p * len(loss)), "Kupiec p": p_k, "binomial p": binomial_pvalue(hits, p),
                         "consecutive": n11, "Christoffersen p": p_c, "Z2": z2, "Z2 crit": z2_crit[q],
                         "pass all": (p_k > 0.05) and (p_c > 0.05) and (z2 > z2_crit[q])})
    return pd.DataFrame(rows).set_index(["level", "method"])


def traffic_light(hits, window=250):
    """Basel traffic light (Equation 25): rolling count of 99% VaR breaches over ``window`` days and its zone.

    Returns
    -------
    pd.DataFrame
        Columns 'breaches' (rolling count, from the window-th day) and 'zone' (green 0-4, yellow 5-9, red 10+).
    """
    count = pd.Series(hits).astype(int).rolling(window).sum()
    zone = pd.cut(count, bins=[-1, 4, 9, np.inf], labels=["green", "yellow", "red"])
    return pd.DataFrame({"breaches": count, "zone": zone})