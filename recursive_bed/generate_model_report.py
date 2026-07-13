"""
Generate an interactive HTML report of the fitted additive latency model.

For the real submesh, fits the two-way additive model
    rtt(s -> t) ~ base(s, t) + mu_src[s] + mu_dst[t]
(base = geodesic d/100 OR fiber floor) with additive_batch_em, then dumps a
browsable page: pick any source or target node and see
  - its estimated parameters  (mu = per-node overhead, sigma = per-node noise),
  - every measurement used to estimate them, with the base term, the model
    prediction, and the residual (observed - predicted),
  - for a target: its estimated vs true location and the error.
Both the geodesic and fiber fits are embedded; a toggle switches between them.

Output: model_report.html  (open in a browser)
"""
from __future__ import annotations
# --- path bootstrap (moved into recursive_bed/): make repo-root modules,
# internet_gmaps, and relative data/figure paths resolve regardless of CWD ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'internet_gmaps'))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT)
# --- end bootstrap ---

import os
import sys
import json
import math

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'internet_gmaps'))

import experiment_audit_vs_task_real as E
from experiment_audit_vs_task_fiber import make_fiber
from probabilistic_helpers import (additive_batch_em, ADDITIVE_PRIOR_MU_MS,
                                    ADDITIVE_PRIOR_VAR_MS2)
from utils import get_distance

KM_PER_MS = 100.0


def base_ms(model, a_loc, b_loc):
    return get_distance(a_loc, b_loc) / KM_PER_MS if model is None else model.base_ms(a_loc, b_loc)


def fit_model(mesh, model):
    src_loc, rtt = mesh['src_loc'], mesh['rtt']
    pairs = {(s, t): [rtt(s, t)] for (s, t) in mesh['task_edges']}
    est, mu_s, var_s, mu_t, var_t = additive_batch_em(pairs, src_loc, n_iters=6,
                                                      rtt_model=model)

    sources, targets = {}, {}
    for s in mesh['sources']:
        meas = []
        for (s2, t), rs in pairs.items():
            if s2 != s:
                continue
            r = rs[0]
            base = base_ms(model, src_loc[s], est.get(t, mesh['tgt_loc'][t]))
            off = mu_s.get(s, ADDITIVE_PRIOR_MU_MS) + mu_t.get(t, ADDITIVE_PRIOR_MU_MS)
            meas.append(dict(other=t, rtt=round(r, 2), base=round(base, 2),
                             pred=round(base + off, 2), resid=round(r - base - off, 2),
                             other_mu=round(mu_t.get(t, ADDITIVE_PRIOR_MU_MS), 2)))
        sources[s] = dict(
            loc=[round(src_loc[s][0], 4), round(src_loc[s][1], 4)],
            mu=round(mu_s.get(s, ADDITIVE_PRIOR_MU_MS), 3),
            sigma=round(math.sqrt(var_s.get(s, ADDITIVE_PRIOR_VAR_MS2)), 3),
            n=len(meas), meas=sorted(meas, key=lambda m: m['rtt']))

    for t in mesh['targets']:
        e = est.get(t, mesh['tgt_loc'][t])
        err = get_distance(e, mesh['tgt_loc'][t])
        meas = []
        for (s, t2), rs in pairs.items():
            if t2 != t:
                continue
            r = rs[0]
            base = base_ms(model, src_loc[s], e)
            off = mu_s.get(s, ADDITIVE_PRIOR_MU_MS) + mu_t.get(t, ADDITIVE_PRIOR_MU_MS)
            meas.append(dict(other=s, rtt=round(r, 2), base=round(base, 2),
                             pred=round(base + off, 2), resid=round(r - base - off, 2),
                             other_mu=round(mu_s.get(s, ADDITIVE_PRIOR_MU_MS), 2)))
        targets[t] = dict(
            true=[round(mesh['tgt_loc'][t][0], 4), round(mesh['tgt_loc'][t][1], 4)],
            est=[round(e[0], 4), round(e[1], 4)], err_km=round(err, 1),
            mu=round(mu_t.get(t, ADDITIVE_PRIOR_MU_MS), 3),
            sigma=round(math.sqrt(var_t.get(t, ADDITIVE_PRIOR_VAR_MS2)), 3),
            n=len(meas), meas=sorted(meas, key=lambda m: m['rtt']))

    errs = [targets[t]['err_km'] for t in targets]
    summary = dict(
        n_src=len(sources), n_tgt=len(targets),
        n_meas=len(pairs),
        mean_err=round(sum(errs) / len(errs), 1),
        median_err=round(sorted(errs)[len(errs) // 2], 1),
        mu_src_range=[round(min(v['mu'] for v in sources.values()), 2),
                      round(max(v['mu'] for v in sources.values()), 2)],
        mu_dst_range=[round(min(v['mu'] for v in targets.values()), 2),
                      round(max(v['mu'] for v in targets.values()), 2)])
    return dict(sources=sources, targets=targets, summary=summary)


HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Additive latency model — fitted parameters</title>
<style>
 :root {{ --bg:#0f1720; --panel:#182430; --ink:#e6edf3; --mut:#8b98a5;
          --accent:#2e86ab; --warn:#d1495b; --ok:#6a994e; --line:#2a3b4a; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--ink); }}
 header {{ padding:14px 20px; background:var(--panel); border-bottom:1px solid var(--line); }}
 h1 {{ font-size:17px; margin:0 0 6px; }}
 .sub {{ color:var(--mut); font-size:12.5px; }}
 .controls {{ margin-top:10px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }}
 .controls label {{ cursor:pointer; }}
 .layout {{ display:flex; height:calc(100vh - 118px); }}
 .list {{ width:300px; border-right:1px solid var(--line); overflow:auto; background:#131d27; }}
 .list input {{ width:100%; padding:8px 10px; background:#0d151d; color:var(--ink);
                border:none; border-bottom:1px solid var(--line); outline:none; }}
 .node {{ padding:7px 12px; border-bottom:1px solid #1c2833; cursor:pointer;
          display:flex; justify-content:space-between; gap:8px; }}
 .node:hover {{ background:#1c2b38; }}
 .node.sel {{ background:var(--accent); color:#fff; }}
 .node .id {{ font-family:ui-monospace,monospace; font-size:12px; }}
 .node .tag {{ font-size:10px; padding:1px 6px; border-radius:8px; background:#24384a; color:var(--mut); }}
 .node .tag.src {{ background:#1d3a2e; color:#a7d7b0; }}
 .node .tag.tgt {{ background:#3a2a1d; color:#e6c79a; }}
 .detail {{ flex:1; overflow:auto; padding:20px 26px; }}
 .cards {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:18px; }}
 .card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
          padding:12px 16px; min-width:120px; }}
 .card .k {{ color:var(--mut); font-size:11.5px; text-transform:uppercase; letter-spacing:.04em; }}
 .card .v {{ font-size:20px; font-weight:600; margin-top:2px; font-family:ui-monospace,monospace; }}
 table {{ border-collapse:collapse; width:100%; font-size:13px; }}
 th,td {{ text-align:right; padding:6px 10px; border-bottom:1px solid var(--line); font-family:ui-monospace,monospace; }}
 th {{ color:var(--mut); font-weight:600; position:sticky; top:0; background:var(--bg); text-transform:uppercase; font-size:11px; }}
 td.l,th.l {{ text-align:left; }}
 .resid-hi {{ color:var(--warn); }} .resid-lo {{ color:var(--ok); }}
 .eq {{ font-family:ui-monospace,monospace; color:var(--mut); background:#0d151d;
        padding:8px 12px; border-radius:8px; margin:0 0 16px; font-size:12.5px; }}
 .hint {{ color:var(--mut); font-size:12px; margin-top:4px; }}
 .mapbox {{ display:inline-block; background:#0b1219; border:1px solid var(--line);
            border-radius:10px; padding:8px 10px; margin:0 0 16px; }}
 .mapbox .lg {{ color:var(--mut); font-size:11.5px; margin-top:6px; }}
 .pin-t {{ color:var(--warn); }} .pin-e {{ color:#a679e8; }}
 a {{ color:var(--accent); }}
</style></head><body>
<header>
 <h1>Additive latency model — fitted parameters &amp; the measurements behind them</h1>
 <div class="sub">rtt(src&rarr;dst) &asymp; base(src,dst) + &mu;<sub>src</sub> + &mu;<sub>dst</sub> &nbsp;·&nbsp;
   base = geodesic d/100 or fiber floor · &mu; = per-node overhead (ms), &sigma; = per-node noise (ms) ·
   fit by additive_batch_em (self-supervised, no ground truth)</div>
 <div class="controls">
   <span>Base model:</span>
   <label><input type="radio" name="model" value="geodesic" checked> Geodesic</label>
   <label><input type="radio" name="model" value="fiber"> Fiber</label>
   <span id="summary" class="sub"></span>
 </div>
</header>
<div class="layout">
 <div class="list">
   <input id="filter" placeholder="filter nodes (id)…">
   <div id="nodes"></div>
 </div>
 <div class="detail" id="detail"></div>
</div>
<script>
const DATA = __DATA__;
let model = 'geodesic', sel = null;

function fmtLoc(l){{ return l[0].toFixed(3)+', '+l[1].toFixed(3); }}
function residClass(r){{ return Math.abs(r)>15 ? 'resid-hi' : (Math.abs(r)<5?'resid-lo':''); }}

function renderList(){{
  const d = DATA[model], f = document.getElementById('filter').value.toLowerCase();
  const items = [];
  for (const [id,v] of Object.entries(d.sources))
    items.push({{id, type:'src', mu:v.mu, n:v.n}});
  for (const [id,v] of Object.entries(d.targets))
    items.push({{id, type:'tgt', mu:v.mu, n:v.n, err:v.err_km}});
  const html = items.filter(i=>i.id.toLowerCase().includes(f)).map(i=>
    `<div class="node ${{sel===i.type+':'+i.id?'sel':''}}" onclick="select('${{i.type}}','${{i.id}}')">
       <span class="id">${{i.id}}</span>
       <span><span class="tag ${{i.type}}">${{i.type==='src'?'source':'target'}}</span>
       <span style="color:var(--mut)">&mu;=${{i.mu}}${{i.err!=null?' · '+i.err+'km':''}}</span></span>
     </div>`).join('');
  document.getElementById('nodes').innerHTML = html;
  const s = d.summary;
  document.getElementById('summary').textContent =
    `${{s.n_src}} sources · ${{s.n_tgt}} targets · ${{s.n_meas}} measurements · `+
    `mean err ${{s.mean_err}} km (median ${{s.median_err}}) · `+
    `μ_src∈[${{s.mu_src_range[0]}},${{s.mu_src_range[1]}}] μ_dst∈[${{s.mu_dst_range[0]}},${{s.mu_dst_range[1]}}]`;
}}

function targetMap(node){{
  const b = DATA.bounds, W = 320, pad = 9;
  const mlat = (b[0] + b[1]) / 2 * Math.PI / 180;
  const H = Math.max(150, Math.round(W * (b[1] - b[0]) / ((b[3] - b[2]) * Math.cos(mlat))));
  const proj = (lat, lon) => [pad + (lon - b[2]) / (b[3] - b[2]) * (W - 2 * pad),
                              pad + (1 - (lat - b[0]) / (b[1] - b[0])) * (H - 2 * pad)];
  const dots = DATA.src_locs.map(l => {{ const p = proj(l[0], l[1]);
     return `<circle cx="${{p[0].toFixed(1)}}" cy="${{p[1].toFixed(1)}}" r="1.6" fill="#3b4b5a"/>`; }}).join('');
  const t = proj(node.true[0], node.true[1]), e = proj(node.est[0], node.est[1]);
  return `<div class="mapbox"><svg width="${{W}}" height="${{H}}">
     <rect x="0" y="0" width="${{W}}" height="${{H}}" rx="6" fill="#0b1219"/>
     ${{dots}}
     <line x1="${{t[0].toFixed(1)}}" y1="${{t[1].toFixed(1)}}" x2="${{e[0].toFixed(1)}}" y2="${{e[1].toFixed(1)}}"
           stroke="#f4a259" stroke-width="1.6" stroke-dasharray="4 2"/>
     <circle cx="${{e[0].toFixed(1)}}" cy="${{e[1].toFixed(1)}}" r="6" fill="#8338ec" stroke="#fff" stroke-width="1.4"/>
     <circle cx="${{t[0].toFixed(1)}}" cy="${{t[1].toFixed(1)}}" r="6" fill="#e63946" stroke="#fff" stroke-width="1.4"/>
   </svg>
   <div class="lg"><span class="pin-t">&#9679; true</span> &nbsp; <span class="pin-e">&#9679; estimated</span>
     &nbsp; <span style="color:#6b7a88">&middot; sources</span>
     &nbsp; &mdash; error <b>${{node.err_km}} km</b></div></div>`;
}}

function measTable(meas, otherLabel, otherMuLabel){{
  const rows = meas.map(m=>`<tr>
     <td class="l">${{m.other}}</td>
     <td>${{m.other_mu}}</td>
     <td>${{m.rtt}}</td>
     <td>${{m.base}}</td>
     <td>${{m.pred}}</td>
     <td class="${{residClass(m.resid)}}">${{m.resid>0?'+':''}}${{m.resid}}</td>
   </tr>`).join('');
  return `<table><thead><tr>
     <th class="l">${{otherLabel}}</th><th>${{otherMuLabel}} (ms)</th>
     <th>rtt (ms)</th><th>base (ms)</th><th>pred (ms)</th><th>residual (ms)</th>
   </tr></thead><tbody>${{rows}}</tbody></table>`;
}}

function select(type,id){{
  sel = type+':'+id;
  const d = DATA[model];
  const node = (type==='src'?d.sources:d.targets)[id];
  let cards, eq, table;
  if(type==='src'){{
    cards = [['role','source'],['location',fmtLoc(node.loc)],
             ['μ_src (overhead ms)',node.mu],['σ_src (noise ms)',node.sigma],
             ['# measurements',node.n]];
    eq = 'This node acts as a vantage point. Its μ_src is fit as the shrunk mean of '+
         '(rtt − base − μ_dst) over all destinations below. Residual = rtt − (base + μ_src + μ_dst).';
    table = measTable(node.meas,'destination','μ_dst');
  }} else {{
    cards = [['role','target (unknown IP)'],['true location',fmtLoc(node.true)],
             ['estimated location',fmtLoc(node.est)],['error (km)',node.err_km],
             ['μ_dst (overhead ms)',node.mu],['σ_dst (noise ms)',node.sigma],
             ['# measurements',node.n]];
    eq = 'This node is geolocated. Its location (MAP) and μ_dst are fit jointly from the '+
         'sources below; μ_dst is the shrunk mean of (rtt − base − μ_src). Residual = rtt − (base + μ_src + μ_dst).';
    table = measTable(node.meas,'source','μ_src');
  }}
  document.getElementById('detail').innerHTML =
    `<div class="cards">${{cards.map(c=>`<div class="card"><div class="k">${{c[0]}}</div><div class="v">${{c[1]}}</div></div>`).join('')}}</div>`+
    (type==='tgt'?targetMap(node):'')+
    `<div class="eq">${{eq}}</div>`+
    `<div class="hint">Sorted by RTT. Residuals |&gt;15 ms| in red, |&lt;5 ms| in green.</div>`+table;
  renderList();
}}

document.querySelectorAll('input[name=model]').forEach(r=>r.onchange=e=>{{
  model=e.target.value; sel=null;
  document.getElementById('detail').innerHTML='<p class="hint">Select a node on the left.</p>';
  renderList();
}});
document.getElementById('filter').oninput=renderList;
renderList();
document.getElementById('detail').innerHTML='<p class="hint">Select a source or target on the left to see its fitted parameters and the measurements behind them.</p>';
</script></body></html>"""


def main():
    mesh = E.load_submesh()
    fiber = make_fiber(mesh)
    print("fitting geodesic model...")
    geo = fit_model(mesh, None)
    print("fitting fiber model...")
    fib = fit_model(mesh, fiber)
    src_locs = [[round(mesh['src_loc'][s][0], 4), round(mesh['src_loc'][s][1], 4)]
                for s in mesh['sources']]
    lats = [l[0] for l in src_locs] + [geo['targets'][t]['true'][0] for t in geo['targets']]
    lons = [l[1] for l in src_locs] + [geo['targets'][t]['true'][1] for t in geo['targets']]
    bounds = [round(min(lats) - 1, 3), round(max(lats) + 1, 3),
              round(min(lons) - 1, 3), round(max(lons) + 1, 3)]
    data = {"geodesic": geo, "fiber": fib, "src_locs": src_locs, "bounds": bounds}
    # The template uses {{ }} for literal braces; de-double them BEFORE
    # injecting the JSON (whose own braces must stay intact).
    html = HTML.replace("{{", "{").replace("}}", "}").replace("__DATA__", json.dumps(data))
    out = "model_report.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out}  (open in a browser)")
    print(f"  geodesic: mean err {geo['summary']['mean_err']} km, "
          f"fiber: mean err {fib['summary']['mean_err']} km")


if __name__ == "__main__":
    main()
