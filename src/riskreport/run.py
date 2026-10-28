"""
The whole evening in one command: data -> fund -> models -> risk -> backtest -> limits -> HTML report.

    python -m riskreport.run                      # download fresh data, report on the latest close
    python -m riskreport.run --no-download        # reuse the raw files of the last download
    python -m riskreport.run --date 2026-10-27    # report as of an earlier close (data cut at that date)

This is what GitHub Actions runs every evening (Step 11). Every number comes from config/config.yaml.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # no screen on a server: draw charts in memory
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from .config import asset_table, load_config
from .daily import evening_risk, var_of_positions
from .data import (build_eurusd, eur_from_factors, factor_returns, load_fred, load_yahoo, mask_stale,
                   quality_checks)
from .engine import _put_state, rolling_risk
from .limits import commentary, limit_table, vol_drivers
from .plots import plot_risk_budget, plot_var_backtest
from .portfolio import fund_history, put_market
from .report import fig_to_base64, render_report, save_report
from .risk import compound_paths, euler_decomposition, scenario_pnl, simulate_paths
from .stress import hypothetical_moves, window_factor_moves

__all__ = ["main", "load_data", "prepare", "update_backtest", "backtest_summary", "build_context"]


def log(msg, t0=[time.time()]):
    """Print a progress line with the seconds elapsed since the start."""
    print(f"[{time.time() - t0[0]:6.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 1. Data                                                                     #
# --------------------------------------------------------------------------- #
def load_data(cfg, raw_dir, download=True):
    """Yahoo prices (assets, market, stress proxies) and FRED series; saved to / read from ``raw_dir``."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    f_y, f_f = raw_dir / "yahoo_prices.csv", raw_dir / "fred_series.csv"
    if download:
        tickers = list(asset_table(cfg)["ticker"]) + list(cfg["market"].values()) \
            + [p["ticker"] for p in cfg["stress"]["proxies"].values()]
        tickers = list(dict.fromkeys(tickers))                           # unique, order kept
        yahoo = load_yahoo(tickers, start=cfg["fund"]["start"])
        fred = pd.concat([load_fred(c, start=cfg["fund"]["start"]) for c in cfg["fred"].values()], axis=1, sort=True)
        yahoo.to_csv(f_y)
        fred.to_csv(f_f)
    else:
        if not (f_y.exists() and f_f.exists()):
            raise FileNotFoundError(f"No raw files in {raw_dir}: run once without --no-download")
        yahoo = pd.read_csv(f_y, index_col=0, parse_dates=True)
        fred = pd.read_csv(f_f, index_col=0, parse_dates=True)
    return yahoo, fred


def prepare(cfg, yahoo, fred, date=None):
    """Clean prices, EUR/USD, risk factors, fund history (with the put) - everything up to ``date``.

    Returns
    -------
    dict
        'assets', 'currencies', 'prices' (clean, by asset), 'proxy_prices', 'eurusd', 'provisional', 'vix',
        'fac' (factor returns), 'pm' (put market), 'put_cfg', 'hist' (fund history), 'suspects', 'stale_days'.
    """
    if date is not None:
        yahoo, fred = yahoo.loc[:date], fred.loc[:date]
    assets = asset_table(cfg)
    currencies = assets["currency"]
    mk, fr = cfg["market"], cfg["fred"]

    prices = yahoo[list(assets["ticker"])].copy()
    prices.columns = assets.index                                        # tickers -> asset names
    exchanges = pd.Series(np.where(assets["ticker"].str.endswith(".DE"), "Xetra", "US"), index=assets.index)
    prices, stale_days = mask_stale(prices, exchanges)

    eurusd, provisional = build_eurusd(fred[fr["eurusd_check"]], yahoo[mk["eurusd"]])
    vix = yahoo[mk["vix"]].dropna()
    fac = factor_returns(prices, eurusd, vix)

    reviewed = cfg.get("data_quality", {}).get("reviewed", [])
    suspects, _ = quality_checks(eur_from_factors(fac, currencies), reviewed=reviewed)

    put_cfg = cfg["overlay"] if cfg["overlay"]["enabled"] else None
    pm = put_market(yahoo[mk["index"]].dropna(), vix, fred[fr["rate_3m"]].dropna(), eurusd)
    put = {**put_cfg, "market": pm} if put_cfg else None
    hist = fund_history(eur_from_factors(fac, currencies), assets["weight"], cfg["fund"]["nav"], put=put)

    proxy_prices = yahoo[[p["ticker"] for p in cfg["stress"]["proxies"].values()]]
    return {"assets": assets, "currencies": currencies, "prices": prices, "proxy_prices": proxy_prices,
            "eurusd": eurusd, "provisional": provisional, "vix": vix, "fac": fac, "pm": pm, "put_cfg": put_cfg,
            "hist": hist, "suspects": suspects, "stale_days": stale_days}


# --------------------------------------------------------------------------- #
# 2. Out-of-sample backtest, extended day by day                              #
# --------------------------------------------------------------------------- #
def _read_oos(path):
    """The saved out-of-sample table (a DataFrame, or the (table, params) pair returned by rolling_risk)."""
    obj = pd.read_pickle(path)
    if isinstance(obj, tuple):
        obj = obj[0]
    elif isinstance(obj, dict):
        obj = obj["out"]
    return obj


def update_backtest(d, cfg, path):
    """Add to the saved backtest the days after its last date (same engine, no look-ahead) and save it.

    If the file does not exist, the full backtest is run (slow: about 10 minutes).
    """
    fac, mc = d["fac"], cfg["models"]
    kw = dict(window=mc["fit_window"], refit_every=20, hs_window=mc["hs_window"],
              levels=cfg["measures"]["var_levels"], pool="rolling", evt_share=mc["evt_tail_share"])
    path = Path(path)
    if path.exists():
        oos = _read_oos(path)
        oos = oos.loc[:fac.index[-1]]                                    # --date in the past: cut
        new = fac.index[fac.index > oos.index[-1]]
        if len(new):
            start = fac.index.get_loc(new[0])
            add, _ = rolling_risk(fac, d["currencies"], d["hist"], d["pm"], d["put_cfg"], start=start, **kw)
            oos = pd.concat([oos, add[oos.columns]])
            log(f"backtest: {len(new)} new day(s) added ({new[0].date()} to {new[-1].date()})")
        else:
            log("backtest: already up to date")
    else:
        log("backtest: no saved file, running the full out-of-sample engine (about 10 minutes)")
        oos, _ = rolling_risk(fac, d["currencies"], d["hist"], d["pm"], d["put_cfg"], **kw)
    if fac.index[-1] >= oos.index[-1]:                                    # never overwrite with a cut history
        oos.to_pickle(path)
    return oos


def backtest_summary(oos, nav, method="FHS + EVT", level=0.99, window=250):
    """Breaches of VaR, Kupiec test (Equation 18 of P3) and Basel traffic light on the last ``window`` days."""
    loss = oos[("loss", "", "")]
    var = oos[(method, "VaR", level)]
    hits = (loss > var).astype(int)
    n, x, p = len(hits), int(hits.sum()), 1 - level
    ph = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - ph) + (x * np.log(ph) if x else 0.0) if 0 < ph < 1 else 0.0
    kupiec_p = float(chi2.sf(-2 * (ll0 - ll1), 1))
    recent = int(hits.iloc[-window:].sum())
    zone = "green" if recent <= 4 else ("yellow" if recent <= 9 else "red")
    nav_prev = nav.shift(1).reindex(oos.index)
    return {"zone": zone, "breaches_250": recent, "breaches_total": x, "expected_total": n * p,
            "kupiec_p": kupiec_p, "since": str(oos.index[0].date()),
            "loss_pct": 100 * loss / nav_prev, "var_pct": 100 * var / nav_prev,
            "last_hit": bool(hits.iloc[-1]), "last_loss": float(loss.iloc[-1]), "last_var": float(var.iloc[-1])}


# --------------------------------------------------------------------------- #
# 3. Tonight's measures                                                       #
# --------------------------------------------------------------------------- #
def ucits_var(ev, d, cfg, level=0.99):
    """20-day VaR 99% (fraction of NAV) from simulated GJR + DCC paths, put revalued at the horizon."""
    m, mc = ev["models"], cfg["models"]
    p = m["params"]
    paths = simulate_paths(m["e"], p["mu"], p["omega"], p["alpha"], p["gamma"], p["beta"], m["sigma_next"],
                           m["Q_next"], m["dcc"]["Qbar"], m["dcc"]["a"], m["dcc"]["b"],
                           horizon=mc["horizon_days"], n_paths=mc["n_paths"], seed=mc["seed"])
    scen = compound_paths(paths, list(d["fac"].columns))
    put = ev["positions"]["put"]
    if put is not None:
        put = {**put, "dt": (ev["date"] + pd.offsets.BDay(mc["horizon_days"]) - ev["date"]).days / 365}
    pos = ev["positions"]
    L = -scenario_pnl(scen, d["currencies"], pos["w"], pos["A"], put)["total"].to_numpy()
    return np.quantile(L, level) / pos["nav"]


def stress_table(ev, d, cfg):
    """Historical and hypothetical stress P&L of tonight's positions, with and without the put (instantaneous)."""
    pos, st = ev["positions"], cfg["stress"]
    put0 = {**pos["put"], "dt": 0.0} if pos["put"] is not None else None
    rows = []

    def pnl(moves, put):
        scen = pd.DataFrame([moves.reindex(d["fac"].columns).fillna(0.0)])
        return float(scenario_pnl(scen, d["currencies"], pos["w"], pos["A"], put)["total"].iloc[0])

    for key, s in st["historical"].items():
        mv, used = window_factor_moves(d["prices"], d["proxy_prices"], st["proxies"], d["eurusd"], d["vix"],
                                       s["start"], s["end"])
        note = f"{s['start']} to {s['end']}" + (f"; proxies: {', '.join(f'{a} = {t}' for a, t in used.items())}"
                                                if used else "")
        rows.append({"key": key, "label": s["label"], "kind": "historical", "moves": mv, "note": note})
    sigma0 = d["vix"].iloc[-1] / 100
    for key, s in st["hypothetical"].items():
        mv = hypothetical_moves(s, d["assets"], sigma0)
        rows.append({"key": key, "label": s["label"], "kind": "hypothetical", "moves": mv, "note": ""})
    for r in rows:
        r["pnl_eur"] = pnl(r["moves"], put0)
        r["pnl_pct"] = 100 * r["pnl_eur"] / pos["nav"]
        r["no_put_pct"] = 100 * pnl(r["moves"], None) / pos["nav"]
    return pd.DataFrame(rows).set_index("key")


def exposures(ev, d):
    """Absolute EUR exposure of the fund to each factor (assets, EUR/USD through the USD assets, VIX = 0)."""
    pos = ev["positions"]
    e = (pos["A"] * pos["w"]).abs()
    usd = d["currencies"].index[d["currencies"] == "USD"]
    return pd.concat([e, pd.Series({"eurusd": e[usd].sum(), "vix": 0.0})])


# --------------------------------------------------------------------------- #
# 4. The report's context                                                     #
# --------------------------------------------------------------------------- #
def build_context(cfg, d, ev, ev_prev, oos, root):
    """Every number, table, chart and sentence of the HTML page (the template's variables)."""
    nav, lim = ev["nav"], cfg["limits"]
    tab = ev["table"]
    main = "FHS + EVT"
    var99, es975 = tab.loc[(main, 0.99), "VaR"], tab.loc[(main, 0.975), "ES"]
    L = -ev["by_pos"].sum(axis=1)
    vol = L.std() * np.sqrt(252) / nav
    ucits = ucits_var(ev, d, cfg)
    log(f"UCITS 20-day VaR 99%: {100 * ucits:.2f}% of NAV")

    dec = euler_decomposition(ev["by_pos"], 0.99, 0.975)
    top = dec.drop(index="put", errors="ignore")["share of VaR"].idxmax()
    st = stress_table(ev, d, cfg)
    worst = st["pnl_pct"].idxmin()
    log(f"stress: worst {worst} {st.loc[worst, 'pnl_pct']:.2f}% of NAV")

    values = {"var_99_1d": var99 / nav, "es_975_1d": es975 / nav, "ucits_var_99_20d": ucits,
              "volatility_annual": vol, "max_risk_contribution": dec.loc[top, "share of VaR"],
              "worst_stress_loss": -st.loc[worst, "pnl_pct"] / 100}
    limits = limit_table(values, lim)

    # Day-over-day change of VaR 99%: positions effect and market effect (Equation 24)
    pos_now, pos_prev = ev["positions"], ev_prev["positions"]
    v_prev = ev_prev["table"].loc[(main, 0.99), "VaR"]
    v_mixed = var_of_positions(ev["scen"], d["currencies"], pos_prev, [0.99]).loc[0.99, "VaR"]
    pos_effect, mkt_effect = var99 - v_mixed, v_mixed - v_prev
    drivers = vol_drivers(pd.Series(ev["models"]["sigma_next"], d["fac"].columns),
                          pd.Series(ev_prev["models"]["sigma_next"], d["fac"].columns), exposures(ev, d))
    readable = {**d["assets"]["name"].to_dict(), "eurusd": "EUR/USD", "vix": "VIX"}
    drivers = drivers.rename(index=readable)                             # names in the commentary, not codes

    bt = backtest_summary(oos, d["hist"]["nav"])
    decision = Path(root) / "data" / "processed" / "model_decision.json"
    mult = json.loads(decision.read_text()).get("var999_multiplier", 1.0) if decision.exists() else 1.0
    prov = d["provisional"]
    prov_dates = [str(x.date()) for x in prov.index[prov] if x <= ev["date"]]
    n_suspects = int((d["suspects"]["status"] == "TO CHECK").sum())

    facts = {"date": ev["date"].date(), "var": var99, "var_prev": v_prev, "pos_effect": pos_effect,
             "mkt_effect": mkt_effect, "nav": nav, "drivers": drivers, "limits": limits,
             "top_contrib": (d["assets"].loc[top, "name"], dec.loc[top, "share of VaR"]),
             "worst_stress": (st.loc[worst, "label"], st.loc[worst, "pnl_pct"]),
             "tl_breaches": bt["breaches_250"], "tl_zone": bt["zone"],
             "breach_yesterday": (bt["last_loss"], bt["last_var"]) if bt["last_hit"] else None,
             "provisional_fx": prov_dates, "data_suspects": n_suspects}

    # Charts (also saved as PNG in reports/figures)
    figs = Path(root) / "reports" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    names = list(d["assets"]["name"]) + ["Protective put"]
    w_nav = pd.concat([pos_now["A"] * pos_now["w"], pd.Series({"put": pos_now["put"]["value0"]
                                                               if pos_now["put"] else 0.0})]) / nav
    fig_b = plot_risk_budget(pd.Series(100 * w_nav.to_numpy(), index=names),
                             pd.Series(100 * dec["share of VaR"].to_numpy(), index=names),
                             title=f"Capital vs risk, {ev['date'].date()}")
    fig_b.savefig(figs / "risk_budget_tonight.png", dpi=110, bbox_inches="tight")
    start_zoom = str(oos.index[-250].date()) if len(oos) > 250 else str(oos.index[0].date())
    fig_t = plot_var_backtest(bt["loss_pct"], {"VaR 99% (FHS + EVT)": bt["var_pct"]},
                              zoom=(start_zoom, str(oos.index[-1].date())),
                              title="Daily loss of the fund vs the VaR forecast of the evening before")
    fig_t.savefig(figs / "var_backtest_tonight.png", dpi=110, bbox_inches="tight")
    charts = {"risk_budget": fig_to_base64(fig_b), "backtest": fig_to_base64(fig_t)}
    plt.close("all")

    levels = cfg["measures"]["var_levels"]
    cols = [f"VaR {100 * q:g}%" for q in levels] + [f"ES {100 * q:g}%" for q in levels]
    measures = [{"method": m, "main": m == main,
                 "values": [100 * tab.loc[(m, q), "VaR"] / nav for q in levels]
                 + [100 * tab.loc[(m, q), "ES"] / nav for q in levels]}
                for m in ["HS", "Normal", "FHS", main]]
    q_top = max(levels)
    measures.append({"method": f"{main} x {mult:.2f} (model-risk add-on, {100 * q_top:g}% only)", "main": False,
                     "values": [100 * tab.loc[(main, q), "VaR"] / nav * (mult if q == q_top else 1) for q in levels]
                     + [100 * tab.loc[(main, q), "ES"] / nav * (mult if q == q_top else 1) for q in levels]})

    decomposition = [{"name": n, "weight": 100 * w_nav.iloc[k], "var_share": 100 * dec["share of VaR"].iloc[k],
                      "es_share": 100 * dec["share of ES"].iloc[k], "ivar": dec["incremental VaR"].iloc[k] / 1e3}
                     for k, n in enumerate(names)]
    stress = [{"label": r["label"], "kind": r["kind"], "pnl_pct": r["pnl_pct"], "no_put_pct": r["no_put_pct"],
               "pnl_eur": r["pnl_eur"], "note": r["note"]} for _, r in st.iterrows()]
    a = d["assets"]
    positions = [{"name": a.loc[k, "name"], "ticker": a.loc[k, "ticker"], "ccy": a.loc[k, "currency"],
                  "cls": a.loc[k, "asset_class"], "weight": 100 * pos_now["A"] * pos_now["w"][k] / nav,
                  "eur": pos_now["A"] * pos_now["w"][k], "target": 100 * a.loc[k, "weight"]} for k in a.index]
    p = pos_now["put"]
    hist = d["hist"]
    i = len(d["fac"]) - 1
    put_desc = (f"{p['units']:,.0f} units of a S&P 500 put, strike {p['K']:,.0f} "
                f"(index {p['S0']:,.0f}), expiry {hist['put_expiry'].iloc[i].date()}, "
                f"value EUR {p['value0'] / 1e6:,.2f}m ({100 * p['value0'] / nav:.2f}% of NAV), "
                f"VIX {100 * p['sigma0']:.1f}.") if p else "none."
    m = ev["models"]
    notes = [f"Data: Yahoo Finance (prices, VIX, S&P 500, EUR/USD fill) and FRED (EUR/USD DEXUSEU, 3-month T-bill); "
             f"factor history {d['fac'].index[0].date()} to {d['fac'].index[-1].date()} ({len(d['fac']):,} days).",
             f"Models re-estimated tonight on {cfg['models']['fit_window']:,} days: GJR-GARCH on 10 factors and DCC "
             f"(a = {m['dcc']['a']:.4f}, b = {m['dcc']['b']:.4f}); all fits converged: {m['converged']}.",
             f"FHS: {len(ev['scen']):,} scenarios (filtered shocks of the window, recoloured with tomorrow's "
             f"volatilities and correlations); EVT: GPD on the worst {100 * cfg['models']['evt_tail_share']:.0f}% "
             f"of scenario losses; the put is revalued in full (Black-Scholes at VIX).",
             f"UCITS VaR: {cfg['models']['n_paths']:,} simulated {cfg['models']['horizon_days']}-day paths "
             f"(GJR + DCC dynamics, bootstrap of filtered shocks).",
             f"VaR 99.9% multiplied by {mult:.2f}: model-risk add-on set in the backtest (in-sample choice).",
             "Stress tests are instantaneous (positions frozen, put revalued with 0 days elapsed)."]
    if prov_dates:
        notes.append(f"EUR/USD provisional (Yahoo) for {', '.join(prov_dates)}.")

    kpis = [{"label": "VaR 99%, 1 day (% NAV)", "value": f"{100 * var99 / nav:.2f}%"},
            {"label": "VaR 99%, 1 day (EUR)", "value": f"{var99 / 1e6:,.2f}m"},
            {"label": "ES 97.5%, 1 day", "value": f"{100 * es975 / nav:.2f}%"},
            {"label": "UCITS VaR 99%, 20 days", "value": f"{100 * ucits:.2f}%"},
            {"label": "Ex-ante volatility (annual)", "value": f"{100 * vol:.2f}%"},
            {"label": "Worst stress", "value": f"{st.loc[worst, 'pnl_pct']:.1f}%"}]
    limit_rows = [{"label": r["label"], "value": 100 * r["value"], "limit": 100 * r["limit"],
                   "util": 100 * r["utilisation"], "status": r["status"]} for _, r in limits.iterrows()]

    ctx = {"fund_name": cfg["fund"]["name"], "date": str(ev["date"].date()), "nav": nav,
           "base_ccy": cfg["fund"]["base_currency"],
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "kpis": kpis,
           "commentary": commentary(facts), "limits": limit_rows, "measure_cols": cols, "measures": measures,
           "measures_note": "Official measure: FHS + EVT (GJR-GARCH + DCC filtered historical simulation, "
                            "GPD tail). HS = last 500 days; Normal = mean and standard deviation of the FHS scenarios.",
           "decomposition": decomposition, "charts": charts, "stress": stress,
           "backtest": {k: bt[k] for k in ["zone", "breaches_250", "breaches_total", "expected_total",
                                           "kupiec_p", "since"]},
           "positions": positions, "put": {"desc": put_desc}, "notes": notes,
           "repo_url": cfg.get("report", {}).get("repo_url", "")}
    key = {"date": str(ev["date"].date()), "nav": nav, "var99": var99 / nav, "es975": es975 / nav,
           "var999_addon": tab.loc[(main, q_top), "VaR"] / nav * mult, "ucits_20d": ucits, "vol": vol,
           "max_contrib": values["max_risk_contribution"], "max_contrib_asset": top,
           "worst_stress": st.loc[worst, "pnl_pct"] / 100, "tl_zone": bt["zone"], "tl_breaches": bt["breaches_250"],
           "overall": None}
    return ctx, key


# --------------------------------------------------------------------------- #
# 5. Main                                                                     #
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Daily risk report of the multi-asset fund")
    ap.add_argument("--config", default="config/config.yaml", help="path of config.yaml")
    ap.add_argument("--no-download", action="store_true", help="reuse the raw files of the last download")
    ap.add_argument("--date", default=None, help="report as of this close (YYYY-MM-DD); default = latest")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config).resolve()
    root = cfg_path.parents[1]                                           # the project folder
    cfg = load_config(cfg_path)
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    log("1/6 data: " + ("reading saved raw files" if args.no_download else "downloading Yahoo and FRED"))
    yahoo, fred = load_data(cfg, root / "data" / "raw", download=not args.no_download)
    d = prepare(cfg, yahoo, fred, args.date)
    i = len(d["fac"]) - 1
    log(f"2/6 fund: {d['fac'].index[-1].date()}, NAV EUR {d['hist']['nav'].iloc[-1] / 1e6:,.2f}m, "
        f"{len(d['fac']):,} days of factors")

    log("3/6 models: tonight and yesterday evening (GJR + DCC, FHS + EVT)")
    ev = evening_risk(d["fac"], d["currencies"], d["hist"], i, cfg, d["pm"], d["put_cfg"])
    ev_prev = evening_risk(d["fac"], d["currencies"], d["hist"], i - 1, cfg, d["pm"], d["put_cfg"],
                           prev=ev["models"])

    log("4/6 backtest update")
    oos = update_backtest(d, cfg, processed / "oos_rolling.pkl")

    log("5/6 measures, stress tests, limits, commentary")
    ctx, key = build_context(cfg, d, ev, ev_prev, oos, root)

    log("6/6 HTML report")
    html = render_report(ctx)
    dated, index = save_report(html, root / "docs" / "reports", ctx["date"])
    key["overall"] = max([r["status"] for r in ctx["limits"]] + [ctx["backtest"]["zone"]],
                         key=lambda s: {"green": 0, "amber": 1, "yellow": 1, "red": 2}[s])
    hist_file = processed / "daily_history.csv"
    daily = pd.read_csv(hist_file, index_col="date") if hist_file.exists() else pd.DataFrame()
    daily = pd.concat([daily.drop(index=key["date"], errors="ignore"), pd.DataFrame([key]).set_index("date")])
    daily.sort_index().to_csv(hist_file)

    print("\n".join(["", f"Report of {ctx['date']} - overall status {key['overall'].upper()}"]
                    + [f"  - {s}" for s in ctx["commentary"]]
                    + [f"Saved: {dated} ({dated.stat().st_size / 1e3:,.0f} kB) and {index}"]))
    return ctx


if __name__ == "__main__":
    main()
