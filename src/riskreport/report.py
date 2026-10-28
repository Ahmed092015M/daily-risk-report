"""
The daily HTML report: one self-contained page (charts embedded) built from the evening's results
with a Jinja2 template, saved as docs/reports/report_YYYY-MM-DD.html and docs/index.html (GitHub Pages).
"""
import base64
import io
from pathlib import Path

from jinja2 import Environment

__all__ = ["fig_to_base64", "render_report", "save_report"]

STATUS_ORDER = {"green": 0, "amber": 1, "yellow": 1, "red": 2}

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Daily Risk Report - {{ fund_name }} - {{ date }}</title>
<style>
 :root { --ink:#1d2433; --muted:#5b6475; --line:#e3e6ec; --bg:#ffffff; --panel:#f6f7f9;
         --green:#1e8e3e; --amber:#c27c0e; --red:#c5221f; --accent:#1a5fb4; }
 * { box-sizing:border-box; }
 body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 -apple-system,"Segoe UI",Roboto,Arial,sans-serif; }
 .wrap { max-width:1100px; margin:0 auto; padding:24px 16px 48px; }
 header { display:flex; flex-wrap:wrap; justify-content:space-between; gap:12px; border-bottom:2px solid var(--ink); padding-bottom:12px; }
 h1 { font-size:22px; margin:0; } h2 { font-size:17px; margin:28px 0 10px; border-bottom:1px solid var(--line); padding-bottom:4px; }
 .sub { color:var(--muted); font-size:13px; }
 .kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:16px 0; }
 .kpi { background:var(--panel); border-radius:6px; padding:10px 12px; border-left:4px solid var(--accent); }
 .kpi .v { font-size:20px; font-weight:600; } .kpi .l { color:var(--muted); font-size:12px; }
 .pill { display:inline-block; padding:1px 8px; border-radius:10px; color:#fff; font-size:12px; font-weight:600; text-transform:uppercase; }
 .green { background:var(--green); } .amber, .yellow { background:var(--amber); } .red { background:var(--red); }
 ul.comment { background:var(--panel); border-radius:6px; padding:12px 12px 12px 30px; margin:0; }
 ul.comment li { margin:4px 0; }
 .tbl { overflow-x:auto; }
 table { border-collapse:collapse; width:100%; font-size:13px; }
 th, td { padding:5px 8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
 th:first-child, td:first-child { text-align:left; } th { background:var(--panel); font-weight:600; }
 .bar { background:var(--line); height:8px; border-radius:4px; min-width:80px; position:relative; }
 .bar span { position:absolute; left:0; top:0; height:8px; border-radius:4px; }
 img { max-width:100%; height:auto; border:1px solid var(--line); border-radius:4px; }
 .grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }
 .notes { color:var(--muted); font-size:12px; } .notes li { margin:3px 0; }
 footer { margin-top:32px; color:var(--muted); font-size:12px; border-top:1px solid var(--line); padding-top:8px; }
</style></head><body><div class="wrap">

<header>
 <div><h1>Daily Risk Report - {{ fund_name }}</h1>
  <div class="sub">Close of {{ date }} &middot; NAV EUR {{ "{:,.1f}".format(nav / 1e6) }}m &middot; base currency {{ base_ccy }}</div></div>
 <div class="sub" style="text-align:right">Overall status <span class="pill {{ overall }}">{{ overall }}</span><br>
  Generated {{ generated_at }}</div>
</header>

<div class="kpis">
 {% for k in kpis %}<div class="kpi"><div class="v">{{ k.value }}</div><div class="l">{{ k.label }}</div></div>{% endfor %}
</div>

<h2>1. Commentary</h2>
<ul class="comment">{% for s in commentary %}<li>{{ s }}</li>{% endfor %}</ul>

<h2>2. Limits</h2>
<div class="tbl"><table><tr><th>Measure</th><th>Value</th><th>Limit</th><th>Utilisation</th><th></th><th>Status</th></tr>
{% for r in limits %}<tr><td>{{ r.label }}</td><td>{{ "%.2f"|format(r.value) }}%</td><td>{{ "%.2f"|format(r.limit) }}%</td>
<td>{{ "%.0f"|format(r.util) }}%</td>
<td><div class="bar"><span class="{{ r.status }}" style="width:{{ [r.util, 100]|min }}%"></span></div></td>
<td><span class="pill {{ r.status }}">{{ r.status }}</span></td></tr>{% endfor %}
</table></div>

<h2>3. Risk measures (1 day, % of NAV)</h2>
<div class="tbl"><table><tr><th>Method</th>{% for c in measure_cols %}<th>{{ c }}</th>{% endfor %}</tr>
{% for r in measures %}<tr><td>{{ r.method }}{% if r.main %} <b>(official)</b>{% endif %}</td>
{% for v in r["values"] %}<td>{{ "%.2f"|format(v) }}</td>{% endfor %}</tr>{% endfor %}
</table></div>
<p class="sub">{{ measures_note }}</p>

<h2>4. Where the risk comes from</h2>
<div class="grid2">
 <div class="tbl"><table><tr><th>Position</th><th>% NAV</th><th>% of VaR 99%</th><th>% of ES 97.5%</th><th>Incremental VaR, EUR k</th></tr>
 {% for r in decomposition %}<tr><td>{{ r.name }}</td><td>{{ "%.1f"|format(r.weight) }}</td><td>{{ "%.1f"|format(r.var_share) }}</td>
 <td>{{ "%.1f"|format(r.es_share) }}</td><td>{{ "{:,.0f}".format(r.ivar) }}</td></tr>{% endfor %}</table></div>
 <div><img src="data:image/png;base64,{{ charts.risk_budget }}" alt="Capital vs risk"></div>
</div>

<h2>5. Stress tests (instantaneous, tonight's positions)</h2>
<div class="tbl"><table><tr><th>Scenario</th><th>Type</th><th>P&amp;L % NAV</th><th>Without put</th><th>P&amp;L EUR m</th><th>Notes</th></tr>
{% for r in stress %}<tr><td>{{ r.label }}</td><td>{{ r.kind }}</td><td>{{ "%.2f"|format(r.pnl_pct) }}</td>
<td>{{ "%.2f"|format(r.no_put_pct) }}</td><td>{{ "{:,.1f}".format(r.pnl_eur / 1e6) }}</td><td style="text-align:left">{{ r.note }}</td></tr>{% endfor %}
</table></div>

<h2>6. Backtest</h2>
<p>Traffic light <span class="pill {{ backtest.zone }}">{{ backtest.zone }}</span> &middot; {{ backtest.breaches_250 }} breaches of VaR 99% in the last 250 days
&middot; since {{ backtest.since }}: {{ backtest.breaches_total }} breaches for {{ "%.1f"|format(backtest.expected_total) }} expected (Kupiec p = {{ "%.2f"|format(backtest.kupiec_p) }}).</p>
<img src="data:image/png;base64,{{ charts.backtest }}" alt="VaR backtest">

<h2>7. Positions</h2>
<div class="tbl"><table><tr><th>Asset</th><th>Ticker</th><th>Currency</th><th>Class</th><th>Weight, % NAV</th><th>EUR m</th><th>Target, % assets</th></tr>
{% for r in positions %}<tr><td>{{ r.name }}</td><td>{{ r.ticker }}</td><td>{{ r.ccy }}</td><td>{{ r.cls }}</td>
<td>{{ "%.2f"|format(r.weight) }}</td><td>{{ "{:,.2f}".format(r.eur / 1e6) }}</td><td>{{ "%.2f"|format(r.target) }}</td></tr>{% endfor %}
</table></div>
<p class="sub">Protective put: {{ put.desc }}</p>

<h2>8. Model and data</h2>
<ul class="notes">{% for n in notes %}<li>{{ n }}</li>{% endfor %}</ul>

<footer>Automated report - methods: GJR-GARCH + DCC filtered historical simulation, EVT tail, full revaluation of the put.
Educational project: not investment advice. Source code: {{ repo_url }}</footer>
</div></body></html>
"""


def fig_to_base64(fig, dpi=110):
    """A matplotlib figure as a base64 PNG string, to embed in the HTML (the page needs no other file)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_report(ctx):
    """Fill the template with the evening's context (a dict; see the notebook, Step 10.1) and return the HTML."""
    ctx = dict(ctx)
    statuses = [r["status"] for r in ctx["limits"]] + [ctx["backtest"]["zone"]]
    ctx["overall"] = max(statuses, key=lambda s: STATUS_ORDER.get(s, 0))
    return Environment(autoescape=True).from_string(TEMPLATE).render(**ctx)


def save_report(html, docs_dir, date):
    """Write report_<date>.html and index.html (the latest report) in ``docs_dir``; return both paths."""
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    dated = docs_dir / f"report_{date}.html"
    dated.write_text(html, encoding="utf-8")
    index = docs_dir.parent / "index.html"
    index.write_text(html, encoding="utf-8")
    return dated, index