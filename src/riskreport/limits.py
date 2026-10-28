"""
Limits, traffic-light statuses (Equation 23) and the automatic commentary of the daily report.
"""
import numpy as np
import pandas as pd

__all__ = ["limit_table", "vol_drivers", "commentary"]

LABELS = {"var_99_1d": "VaR 99%, 1 day", "es_975_1d": "ES 97.5%, 1 day",
          "ucits_var_99_20d": "UCITS VaR 99%, 20 days", "volatility_annual": "Ex-ante volatility (annual)",
          "max_risk_contribution": "Largest single contribution to VaR", "worst_stress_loss": "Worst stress loss"}


def limit_table(values, limits):
    """Utilisation and status of every limit (Equation 23): green < warning level <= amber < 100% <= red.

    Parameters
    ----------
    values : dict
        Measure name -> value as a fraction (of NAV, or a share for 'max_risk_contribution');
        losses are positive numbers.
    limits : dict
        config['limits'] (same names, plus 'warning_level').

    Returns
    -------
    pd.DataFrame
        Index = measure; columns 'label', 'value', 'limit', 'utilisation', 'status'.
    """
    warn = limits["warning_level"]
    rows = {}
    for k, v in values.items():
        lim = limits[k]
        u = v / lim
        rows[k] = {"label": LABELS.get(k, k), "value": v, "limit": lim, "utilisation": u,
                   "status": "red" if u >= 1 else ("amber" if u >= warn else "green")}
    return pd.DataFrame(rows).T


def vol_drivers(sigma_now, sigma_before, exposure, top=3):
    """The factors whose volatility forecast changed most, weighted by the fund's exposure to them.

    Parameters
    ----------
    sigma_now, sigma_before : pd.Series
        Tomorrow's volatility forecast of each factor, tonight and yesterday evening.
    exposure : pd.Series
        Absolute EUR exposure to each factor (used as weight).

    Returns
    -------
    pd.DataFrame
        The ``top`` factors: 'vol change %' (relative change of the forecast) and 'weighted' (change x exposure).
    """
    chg = sigma_now / sigma_before - 1
    weighted = (sigma_now - sigma_before) * exposure.reindex(chg.index).fillna(0)
    order = weighted.abs().sort_values(ascending=False).index[:top]
    return pd.DataFrame({"vol change %": 100 * chg[order], "weighted": weighted[order]})


def commentary(facts):
    """The automatic commentary: a few plain sentences on what changed and what needs attention.

    Parameters
    ----------
    facts : dict
        'var', 'var_prev', 'pos_effect', 'mkt_effect', 'nav', 'drivers' (``vol_drivers`` output),
        'limits' (``limit_table`` output), 'top_contrib' (name, share), 'worst_stress' (label, loss % NAV),
        'tl_breaches', 'tl_zone', 'breach_yesterday' (None or (loss, VaR)), 'provisional_fx' (list of dates),
        'data_suspects' (int).

    Returns
    -------
    list of str
    """
    f = facts
    out = []
    chg = f["var"] - f["var_prev"]
    main = "market moves (volatility and correlation)" if abs(f["mkt_effect"]) >= abs(f["pos_effect"]) \
        else "position changes (drift or trades)"
    drv = ", ".join(f"{k} volatility {v:+.0f}%" for k, v in f["drivers"]["vol change %"].items())
    out.append(f"VaR 99% (1 day) is EUR {f['var'] / 1e6:,.2f}m ({100 * f['var'] / f['nav']:.2f}% of NAV), "
               f"{'up' if chg >= 0 else 'down'} EUR {abs(chg) / 1e3:,.0f}k ({chg / f['var_prev']:+.1%}) since yesterday, "
               f"mainly from {main}: market EUR {f['mkt_effect'] / 1e3:+,.0f}k, positions EUR {f['pos_effect'] / 1e3:+,.0f}k. "
               f"Largest volatility changes: {drv}.")
    lim = f["limits"]
    flagged = lim[lim["status"] != "green"]
    if flagged.empty:
        out.append("All limits are green.")
    for k, r in flagged.iterrows():
        out.append(f"{r['status'].upper()}: {r['label']} at {100 * r['utilisation']:.0f}% of its limit "
                   f"({100 * r['value']:.2f}% vs {100 * r['limit']:.2f}%)"
                   + (f" - {f['top_contrib'][0]} carries {100 * f['top_contrib'][1]:.1f}% of VaR." if k == "max_risk_contribution" else "."))
    out.append(f"Worst stress test: {f['worst_stress'][0]}, {f['worst_stress'][1]:.1f}% of NAV.")
    if f.get("breach_yesterday"):
        loss, var = f["breach_yesterday"]
        out.append(f"VaR BREACH yesterday: loss EUR {loss / 1e6:,.2f}m vs VaR 99% EUR {var / 1e6:,.2f}m.")
    edge = " - one more breach turns it yellow" if f["tl_zone"] == "green" and f["tl_breaches"] == 4 else ""
    out.append(f"Backtest traffic light: {f['tl_zone']} ({int(f['tl_breaches'])} breaches of VaR 99% in the last 250 days{edge}).")
    if f.get("provisional_fx"):
        out.append(f"EUR/USD provisional (Yahoo) for {', '.join(str(d) for d in f['provisional_fx'])}: "
                   "to be replaced by the official FRED rate.")
    if f.get("data_suspects"):
        out.append(f"Data quality: {f['data_suspects']} suspect price(s) to check.")
    return out