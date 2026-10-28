"""
Charts for the daily risk report. Every function draws one figure and returns it,
so the notebook (and the report) can save it before showing it.
"""
import matplotlib.pyplot as plt
import numpy as np

#__all__ = ["plot_risk_history"]
#__all__ = ["plot_risk_history", "plot_risk_budget"]
__all__ = ["plot_risk_history", "plot_risk_budget", "plot_var_backtest"]




def _clean(ax):
    """Remove the top and right frame lines."""
    ax.spines[["top", "right"]].set_visible(False)


def plot_risk_history(vol_ann, corr, fund_vol, title="Risk history of the fund"):
    """Three panels: GJR volatility of selected assets, DCC correlations of key pairs,
    and the fund's ex-ante volatility against its realised volatility.

    Parameters
    ----------
    vol_ann : pd.DataFrame
        Annualised conditional volatility (%) of the assets to show, indexed by date.
    corr : pd.DataFrame
        DCC correlations of the pairs to show, one column per pair.
    fund_vol : pd.DataFrame
        Columns 'ex-ante' and 'realised' (annualised %), indexed by date.
    title : str
        Overall title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    vol_ann.plot(ax=axes[0], lw=1)
    axes[0].set_ylabel("volatility, % a year")
    axes[0].set_title("Conditional volatility (GJR-GARCH)", fontsize=11)

    corr.plot(ax=axes[1], lw=1)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_ylabel("correlation")
    axes[1].set_title("Conditional correlations (DCC)", fontsize=11)

    axes[2].plot(fund_vol.index, fund_vol["realised"], color="#bbbbbb", lw=1, label="realised (21 days, centred later)")
    axes[2].plot(fund_vol.index, fund_vol["ex-ante"], color="#d62728", lw=1.2, label="ex-ante (model, next day)")
    axes[2].set_ylabel("volatility, % a year")
    axes[2].set_title("The fund (assets, target weights): ex-ante vs realised volatility", fontsize=11)
    for ax in axes:
        ax.legend(fontsize=8, ncol=3, loc="upper left")
        _clean(ax)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig

def plot_risk_budget(weights, risk_shares, title="Capital vs risk"):
    """Horizontal bars: share of NAV (capital) and share of VaR (risk) of each position, side by side.

    Parameters
    ----------
    weights : pd.Series
        Share of NAV of each position (%).
    risk_shares : pd.Series
        Share of VaR of each position (%), same index; negative = the position reduces risk.
    title : str
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    order = risk_shares.sort_values().index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(order) + 1.5))
    ax.barh(y + 0.2, weights.reindex(order), height=0.4, color="#9ecae1", label="% of NAV (capital)")
    ax.barh(y - 0.2, risk_shares.reindex(order), height=0.4, color="#d62728", label="% of VaR 99% (risk)")
    for yi, v in zip(y, risk_shares.reindex(order)):
        ax.text(v + (0.5 if v >= 0 else -0.5), yi - 0.2, f"{v:.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_yticks(y, order)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("%")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    _clean(ax)
    fig.tight_layout()
    return fig

def plot_var_backtest(loss_pct, var_lines, zoom=None, title="VaR backtest"):
    """Daily loss of the fund (bars, % of NAV) against VaR forecasts (lines), with breaches marked.

    Parameters
    ----------
    loss_pct : pd.Series
        Realised daily loss in % of NAV (positive = loss).
    var_lines : dict of {label: pd.Series}
        VaR forecasts in % of NAV; breaches of each are marked.
    zoom : tuple (start, end), optional
        A second panel showing only this period.
    title : str
        Overall title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    periods = [(None, None, "Full period")] + ([(zoom[0], zoom[1], f"Zoom {zoom[0]} to {zoom[1]}")] if zoom else [])
    fig, axes = plt.subplots(len(periods), 1, figsize=(12, 4.2 * len(periods)))
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    for ax, (a, b, lab) in zip(np.atleast_1d(axes), periods):
        l = loss_pct.loc[a:b]
        ax.bar(l.index, l.where(l > 0, 0), width=1.0, color="#bbbbbb", label="daily loss")
        for (name, v), c in zip(var_lines.items(), colors):
            v = v.loc[a:b]
            ax.plot(v.index, v, lw=1.0, color=c, label=name)
            hit = l > v
            ax.scatter(l.index[hit], l[hit], s=14, color=c, zorder=3, label=f"breach of {name} ({int(hit.sum())})")
        ax.set_ylabel("% of NAV")
        ax.set_title(lab, fontsize=11)
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        _clean(ax)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig