"""
Scarce VPs + sparse edge targets — the regime where auditing SHOULD pay.

Auditing the model pays only when the auditable component (per-node offsets) is
the binding error AND task pings can't already pin it by pooling. The dense
anchor mesh fails both. This setup flips it:
  - FEW vantage points (~10-15) -> audit pings (VP<->VP) are concentrated, and
    each VP's offset affects many targets.
  - SPARSE targets (each seen by only a handful of VPs) -> low redundancy, so
    task pings can't pool the source offsets well.

Fast vectorized estimator: symmetric per-node additive model
    rtt(a,b) ~ base(a,b) + mu[a] + mu[b]
with a NumPy-vectorized MAP objective (no Python per-constraint loop) and a
bincount coordinate-descent offset fit. Audit pings are VP<->VP (both known),
so they pin mu[VP] with no location ambiguity.

Sweeps the audit fraction at a fixed budget, geodesic vs (gridded) fiber base.
Output: figures/scarce_vp.pdf (+ .png).
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
import pickle
import numpy as np
from scipy.optimize import minimize

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'internet_gmaps'))
from gridded_fiber import GriddedFiberRtt
from utils import get_distance

KM_PER_MS = 100.0
PRIOR_MU = 5.0
PRIOR_STR = 2.0
N_VP = int(sys.argv[1]) if len(sys.argv) > 1 else 12
N_TGT = int(sys.argv[2]) if len(sys.argv) > 2 else 300
MIN_COV = 3
BUDGET_PER_TGT = 8


def hav(lat, lon, vlat, vlon):
    R = 6371.0
    p1 = np.radians(lat); p2 = np.radians(vlat)
    dphi = np.radians(vlat - lat); dl = np.radians(vlon - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


REGION = (36.0, 60.0, -10.0, 28.0)   # Europe: bound geometry so triangulation is possible


def select():
    d = pickle.load(open('cache/real_mini_mesh.pkl', 'rb'))
    loc0 = d['address_to_loc']; meas = d['loc_loc_meas']
    la0, la1, lo0, lo1 = REGION
    loc = {n: v for n, v in loc0.items() if la0 <= v[0] <= la1 and lo0 <= v[1] <= lo1}
    outdeg = {s: sum(1 for t in meas[s] if t in loc) for s in meas if s in loc}
    vps = sorted(outdeg, key=outdeg.get, reverse=True)[:N_VP]
    vset = set(vps)

    def rtt(a, b):
        v = meas.get(a, {}).get(b)
        return float(v[0] if isinstance(v, list) else v) if v is not None else None
    cov = {}
    for v in vps:
        for t in meas.get(v, {}):
            if t in loc and t not in vset and rtt(v, t) is not None:
                cov[t] = cov.get(t, 0) + 1
    # WELL-COVERED targets (geometry OK): seen by >= 70% of the few VPs, so the
    # binding error is the offset/base model (auditable), not the geometry.
    thr = max(MIN_COV, int(round(0.7 * N_VP)))
    cand = [t for t, c in cov.items() if c >= thr]
    cand.sort(key=lambda t: -cov[t])          # descending coverage: best-covered first
    targets = cand[:N_TGT]
    return loc, meas, vps, targets, rtt, cov


def fit_offsets(a_idx, b_idx, e, N, n_iters=8):
    """Symmetric per-node offset fit: e_row ~ mu[a] + mu[b]. Vectorized Jacobi
    coordinate descent with prior shrinkage."""
    mu = np.full(N, PRIOR_MU)
    cnt = np.bincount(a_idx, minlength=N) + np.bincount(b_idx, minlength=N)
    for _ in range(n_iters):
        pa = e - mu[b_idx]; pb = e - mu[a_idx]
        s = (np.bincount(a_idx, weights=pa, minlength=N)
             + np.bincount(b_idx, weights=pb, minlength=N))
        mu = np.maximum(0.0, (PRIOR_STR * PRIOR_MU + s) / (PRIOR_STR + cnt))
    return mu


class Geo:
    """Geodesic base model with the vectorized interface base_rows(loc)->array."""
    def base(self, a_loc, b_loc):
        return get_distance(a_loc, b_loc) / KM_PER_MS


def run(loc, vps, targets, rtt, task_pairs, audit_pairs, fiber=None, n_iters=6):
    """Vectorized additive EM. task_pairs: (vp, tgt). audit_pairs: (vp, vp2)
    both known. Returns {tgt: (lat,lon)} estimates."""
    nodes = list(vps) + list(targets)
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    node_lat = np.array([loc[n][0] for n in nodes])
    node_lon = np.array([loc[n][1] for n in nodes])
    tgt_set = set(targets)

    # rows: a=source(vp), b=other; known_b=True if b is a VP (audit) => location fixed
    rows = []
    for v, t in task_pairs:
        rows.append((idx[v], idx[t], rtt(v, t), False))
    for v, w in audit_pairs:
        r = rtt(v, w)
        if r is not None:
            rows.append((idx[v], idx[w], r, True))
    rows = [r for r in rows if r[2] is not None]
    a_idx = np.array([r[0] for r in rows]); b_idx = np.array([r[1] for r in rows])
    rt = np.array([r[2] for r in rows]); known_b = np.array([r[3] for r in rows])

    # est locations: VPs fixed at truth; targets init to NN (nearest-RTT VP)
    est_lat = node_lat.copy(); est_lon = node_lon.copy()
    # NN init for targets
    for t in targets:
        ti = idx[t]
        mask = b_idx == ti
        if mask.any():
            j = np.argmin(rt[mask]); v = a_idx[mask][j]
            est_lat[ti] = node_lat[v]; est_lon[ti] = node_lon[v]

    def base_all():
        if fiber is None:
            return hav(est_lat[a_idx], est_lon[a_idx], est_lat[b_idx], est_lon[b_idx]) / KM_PER_MS
        # fiber: per row (few unique), interp; vectorize by looping rows (rows are few)
        out = np.empty(len(rows))
        for k in range(len(rows)):
            out[k] = fiber.base_ms((node_lat[a_idx[k]], node_lon[a_idx[k]]),
                                   (est_lat[b_idx[k]], est_lon[b_idx[k]]))
        return out

    # precompute, per target, the rows where it is the (unknown) endpoint b
    tgt_rows = {idx[t]: np.where((b_idx == idx[t]) & (~known_b))[0] for t in targets}

    mu = np.full(N, PRIOR_MU)
    for _ in range(n_iters):
        base = base_all()
        e = rt - base
        mu = fit_offsets(a_idx, b_idx, e, N)
        # E-step: relocate each target (vectorized objective over its rows)
        for ti, ridx in tgt_rows.items():
            if len(ridx) == 0:
                continue
            vlat = node_lat[a_idx[ridx]]; vlon = node_lon[a_idx[ridx]]
            y = rt[ridx] - mu[a_idx[ridx]] - mu[ti]
            def obj(x):
                if fiber is None:
                    b = hav(x[0], x[1], vlat, vlon) / KM_PER_MS
                else:
                    b = np.array(fiber.base_ms_rows(list(zip(vlat, vlon)), (x[0], x[1])))
                r = y - b
                return float(np.dot(r, r))
            res = minimize(obj, [est_lat[ti], est_lon[ti]], method='Nelder-Mead',
                           options={'maxiter': 120, 'xatol': 0.05, 'fatol': 1e-3})
            est_lat[ti], est_lon[ti] = res.x
    return {t: (est_lat[idx[t]], est_lon[idx[t]]) for t in targets}


def mean_err(est, loc):
    e = [get_distance(est[t], loc[t]) for t in est]
    return float(np.mean(e)), float(np.median(e))


def main():
    loc, meas, vps, targets, rtt, cov = select()
    covs = np.array([cov[t] for t in targets])
    print(f"{len(vps)} VPs, {len(targets)} targets; coverage per target "
          f"min/med/max {covs.min()}/{int(np.median(covs))}/{covs.max()}")

    # gridded fiber over the VP+target bbox
    from fiber_graph import FiberGraph
    from floor_query import FloorEstimator
    from glob import glob
    npz = np.load(sorted(glob('internet_gmaps/data/graph_*.npz'))[-1])
    graph = FiberGraph(npz['node_lat'], npz['node_lon'], npz['edge_src'],
                       npz['edge_dst'], npz['edge_rtt_ms'],
                       edge_feature=npz['edge_feature'] if 'edge_feature' in npz else None,
                       feature_names=tuple(npz['feature_names']) if 'feature_names' in npz else ())
    est = FloorEstimator(graph, np.array([loc[v][0] for v in vps]),
                         np.array([loc[v][1] for v in vps]))
    alln = vps + targets
    la = [loc[n][0] for n in alln]; lo = [loc[n][1] for n in alln]
    fiber = GriddedFiberRtt(est, [loc[v] for v in vps],
                            (min(la)-3, max(la)+3, min(lo)-3, max(lo)+3), res_deg=0.4)

    # task order (round-robin, per-seed shuffle) and audit order (VP-VP)
    def task_order(seed):
        rng = np.random.default_rng(seed)
        per = {t: [v for v in vps if rtt(v, t) is not None] for t in targets}
        for t in per:
            rng.shuffle(per[t])
        order = []
        for k in range(len(vps)):
            for t in targets:
                if k < len(per[t]):
                    order.append((per[t][k], t))
        return order

    def audit_order(seed):
        rng = np.random.default_rng(seed + 7)
        pairs = [(vps[i], vps[j]) for i in range(len(vps)) for j in range(len(vps))
                 if i != j and rtt(vps[i], vps[j]) is not None]
        rng.shuffle(pairs)
        return pairs

    budget = BUDGET_PER_TGT * len(targets)
    fractions = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    seeds = list(range(12))
    import time
    res = {'geodesic': [], 'fiber': []}
    for base_name, fib in [('geodesic', None), ('fiber', fiber)]:
        print(f"--- {base_name} base ---")
        for f in fractions:
            errs = []
            for s in seeds:
                to = task_order(s); ao = audit_order(s)
                A = min(int(round(f * budget)), len(ao)); Bt = budget - A
                tp = [to[i % len(to)] for i in range(Bt)] if Bt > 0 else []
                ap = ao[:A]
                t0 = time.time()
                est_locs = run(loc, vps, targets, rtt, tp, ap, fiber=fib)
                errs.append(mean_err(est_locs, loc)[0])
            m = float(np.mean(errs))
            res[base_name].append(m)
            print(f"  audit={int(f*100):>2}%  mean={m:6.0f} km  ({time.time()-t0:.1f}s/seed)")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fp = np.array(fractions) * 100
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.plot(fp, res['geodesic'], '-o', color='#c9a227', lw=2, label='geodesic base')
    ax.plot(fp, res['fiber'], '-o', color='#2e86ab', lw=2, label='fiber base')
    for arr, col in ((res['geodesic'], '#c9a227'), (res['fiber'], '#2e86ab')):
        ax.axvline(fractions[int(np.argmin(arr))] * 100, color=col, ls=':', alpha=0.7)
    ax.set_xlabel('share of budget spent on AUDIT (%)')
    ax.set_ylabel('mean localization error (km)')
    ax.set_title(f'Scarce VPs ({len(vps)}) + sparse targets ({len(targets)}, '
                 f'cov {covs.min()}-{covs.max()}) — does auditing pay?')
    ax.legend(frameon=False, fontsize=10); ax.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs('figures/dynamics', exist_ok=True)
    out = f'figures/dynamics/scarce_vp_{len(vps)}vp.pdf'
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.pdf', '.png'), dpi=140, bbox_inches='tight')
    print(f"wrote {out}")
    print("geodesic:", {int(f*100): round(v) for f, v in zip(fractions, res['geodesic'])})
    print("fiber   :", {int(f*100): round(v) for f, v in zip(fractions, res['fiber'])})


if __name__ == '__main__':
    main()
