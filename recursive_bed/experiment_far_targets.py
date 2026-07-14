"""
Far targets — a real-data regime where auditing DOES pay.

Sources are clustered (Europe); a FEW targets sit in a distant region (e.g.
East Asia / Oceania), reached only over long-haul paths. Two things then hold
that were absent on the dense anchor mesh:
  1. GEOMETRY is one-sided (all sources at ~the same bearing) -> triangulation
     can't resolve distance along the Europe->region axis; the MODEL must carry
     it.
  2. The binding error is a SHARED, region-specific long-haul overhead (the
     "loose floor" / trombone) that, with only a few targets, task pings can't
     pool away -- but an AUDIT to a known landmark IN THAT REGION measures it
     directly and transfers to all targets there.

Model:  rtt(src, dst) = d(src,dst)/100 (or fiber floor) + mu_src + REG + mu_node
where REG is a single shared scalar for the far region (pinned by auditing any
known landmark in it). Estimator is vectorized EM (fast). We sweep the audit
fraction and report far-target localization error, geodesic vs fiber base.

Output: figures/scale/far_targets_<region>.pdf   (see RESULTS.md)
"""
from __future__ import annotations
# --- path bootstrap (recursive_bed/) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, 'internet_gmaps'))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_ROOT)
# --- end bootstrap ---

import os
import pickle
import numpy as np
from scipy.optimize import minimize
from glob import glob

from gridded_fiber import GriddedFiberRtt
from utils import get_distance

KM_PER_MS = 100.0
PRIOR_MU = 0.0          # per-node deviation shrinks to 0 so REG carries the common part
PRIOR_STR_NODE = 1.0
SRC_BOX = (40.0, 55.0, 0.0, 20.0)          # Europe sources
REGIONS = {'easia': (20, 45, 100, 145), 'oceania': (-45, -10, 110, 180),
           'samerica': (-40, 10, -80, -35), 'wus': (30, 50, -125, -100)}
REGION = _sys.argv[1] if len(_sys.argv) > 1 else 'easia'
N_SRC = int(_sys.argv[2]) if len(_sys.argv) > 2 else 25
N_TGT = int(_sys.argv[3]) if len(_sys.argv) > 3 else 6


def hav(lat, lon, vlat, vlon):
    R = 6371.0
    p1 = np.radians(lat); p2 = np.radians(vlat)
    dphi = np.radians(vlat - lat); dl = np.radians(vlon - lon)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))


def load():
    d = pickle.load(open('cache/real_mini_mesh.pkl', 'rb'))
    loc = d['address_to_loc']; meas = d['loc_loc_meas']
    def rtt(a, b):
        v = meas.get(a, {}).get(b)
        return float(v[0] if isinstance(v, list) else v) if v is not None else None
    def box(bx):
        la0, la1, lo0, lo1 = bx
        return {n for n, (la, lo) in loc.items() if la0 <= la <= la1 and lo0 <= lo <= lo1}
    outdeg = {s: sum(1 for t in meas[s] if t in loc) for s in meas if s in loc}
    srcs = sorted([s for s in box(SRC_BOX) if s in outdeg], key=outdeg.get, reverse=True)[:N_SRC]
    far = [t for t in box(REGIONS[REGION])
           if sum(1 for s in srcs if rtt(s, t) is not None) >= N_SRC * 0.7]
    return loc, srcs, far, rtt


def estimate(loc, srcs, targets, landmarks, rtt, task_pairs, audit_pairs, fiber=None, n_iters=8):
    """Vectorized EM with a shared far-region scalar REG. Returns target est locs."""
    src_loc = {s: loc[s] for s in srcs}
    si = {s: i for i, s in enumerate(srcs)}
    slat = np.array([loc[s][0] for s in srcs]); slon = np.array([loc[s][1] for s in srcs])
    far_nodes = list(targets) + list(landmarks)
    fi = {n: i for i, n in enumerate(far_nodes)}
    known = np.array([n in set(landmarks) for n in far_nodes])   # landmark locations known
    est = {n: (loc[n] if n in set(landmarks) else None) for n in far_nodes}
    # NN + region-centroid init for targets
    lc = (np.mean([loc[l][0] for l in landmarks]), np.mean([loc[l][1] for l in landmarks])) \
        if landmarks else (np.mean([loc[t][0] for t in targets]), np.mean([loc[t][1] for t in targets]))
    for t in targets:
        est[t] = lc

    rows = [(si[s], fi[t], rtt(s, t), False) for s, t in task_pairs if rtt(s, t) is not None]
    rows += [(si[s], fi[l], rtt(s, l), True) for s, l in audit_pairs if rtt(s, l) is not None]
    a_idx = np.array([r[0] for r in rows]); b_idx = np.array([r[1] for r in rows])
    rt = np.array([r[2] for r in rows])
    ns, nf = len(srcs), len(far_nodes)

    def base_rows():
        blat = np.array([est[far_nodes[j]][0] for j in b_idx])
        blon = np.array([est[far_nodes[j]][1] for j in b_idx])
        if fiber is None:
            return hav(slat[a_idx], slon[a_idx], blat, blon) / KM_PER_MS
        out = np.empty(len(rows))
        for k in range(len(rows)):
            out[k] = fiber.base_ms((slat[a_idx[k]], slon[a_idx[k]]), (blat[k], blon[k]))
        return out

    mu_s = np.zeros(ns); reg = 0.0; mu_n = np.zeros(nf)
    tgt_rows = {fi[t]: np.where(b_idx == fi[t])[0] for t in targets}
    for _ in range(n_iters):
        e = rt - base_rows()
        # M-step: coordinate descent for mu_s, reg (shared), mu_n (shrunk to 0)
        for _ in range(6):
            # mu_s
            adj = e - reg - mu_n[b_idx]
            mu_s = np.bincount(a_idx, adj, ns) / np.maximum(1, np.bincount(a_idx, minlength=ns))
            # reg (shared over all far rows)
            reg = float(np.mean(e - mu_s[a_idx] - mu_n[b_idx]))
            # mu_n (per far node, shrunk toward 0)
            adj = e - mu_s[a_idx] - reg
            s = np.bincount(b_idx, adj, nf); c = np.bincount(b_idx, minlength=nf)
            mu_n = s / (c + PRIOR_STR_NODE)
        # E-step: relocate targets (reg + mu_s + mu_n known)
        for j, ridx in tgt_rows.items():
            if len(ridx) == 0:
                continue
            vlat = slat[a_idx[ridx]]; vlon = slon[a_idx[ridx]]
            y = rt[ridx] - mu_s[a_idx[ridx]] - reg - mu_n[j]
            def obj(x):
                b = (hav(x[0], x[1], vlat, vlon) / KM_PER_MS) if fiber is None \
                    else np.array(fiber.base_ms_rows(list(zip(vlat, vlon)), (x[0], x[1])))
                r = y - b
                return float(np.dot(r, r))
            x0 = est[far_nodes[j]]
            res = minimize(obj, [x0[0], x0[1]], method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 0.05, 'fatol': 1e-3})
            est[far_nodes[j]] = (float(res.x[0]), float(res.x[1]))
    return {t: est[t] for t in targets}


def main():
    loc, srcs, far, rtt = load()
    rng = np.random.default_rng(0)
    print(f"region={REGION}: {len(srcs)} EU sources, {len(far)} far anchors")
    if len(far) < N_TGT + 5:
        print("not enough far anchors"); return

    # gridded fiber over the whole span (sources + far region)
    npz = np.load(sorted(glob('internet_gmaps/data/graph_*.npz'))[-1])
    from fiber_graph import FiberGraph
    from floor_query import FloorEstimator
    graph = FiberGraph(npz['node_lat'], npz['node_lon'], npz['edge_src'], npz['edge_dst'],
                       npz['edge_rtt_ms'],
                       edge_feature=npz['edge_feature'] if 'edge_feature' in npz else None,
                       feature_names=tuple(npz['feature_names']) if 'feature_names' in npz else ())
    est_f = FloorEstimator(graph, np.array([loc[s][0] for s in srcs]),
                           np.array([loc[s][1] for s in srcs]))
    alln = [loc[n] for n in srcs + far]
    la = [p[0] for p in alln]; lo = [p[1] for p in alln]
    fiber = GriddedFiberRtt(est_f, [loc[s] for s in srcs],
                            (min(la)-5, max(la)+5, min(lo)-5, max(lo)+5), res_deg=0.5)

    fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    seeds = list(range(8))
    budget_per_tgt = 12
    res = {'geodesic': [], 'fiber': []}
    for base_name, fib in [('geodesic', None), ('fiber', fiber)]:
        for f in fractions:
            errs = []
            for s in seeds:
                r = np.random.default_rng(s)
                perm = list(far); r.shuffle(perm)
                targets = perm[:N_TGT]; landmarks = perm[N_TGT:]
                budget = budget_per_tgt * N_TGT
                # task pairs: round-robin sources per target
                srcs_sh = list(srcs);
                tp = [(srcs_sh[i % len(srcs_sh)], targets[(i//1) % N_TGT]) for i in range(budget)]
                # audit pairs: source -> landmark (both known)
                ap_all = [(sx, l) for l in landmarks for sx in srcs if rtt(sx, l) is not None]
                r.shuffle(ap_all)
                A = min(int(round(f * budget)), len(ap_all)); Bt = budget - A
                tp = tp[:Bt]; ap = ap_all[:A]
                est_locs = estimate(loc, srcs, targets, landmarks, rtt, tp, ap, fiber=fib)
                errs.append(np.mean([get_distance(est_locs[t], loc[t]) for t in targets]))
            res[base_name].append(float(np.mean(errs)))
        print(f"{base_name}:", {int(f*100): round(v) for f, v in zip(fractions, res[base_name])})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fp = np.array(fractions) * 100
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.plot(fp, res['geodesic'], '-o', color='#c9a227', lw=2, label='geodesic base')
    ax.plot(fp, res['fiber'], '-o', color='#2e86ab', lw=2, label='fiber base')
    for arr, col in ((res['geodesic'], '#c9a227'), (res['fiber'], '#2e86ab')):
        ax.axvline(fractions[int(np.argmin(arr))]*100, color=col, ls=':', alpha=0.7)
    ax.set_xlabel('share of budget spent on AUDIT (landmark in far region) (%)')
    ax.set_ylabel(f'mean localization error of far targets (km)')
    ax.set_title(f'Far targets ({REGION}, {N_TGT} tgts) via {len(srcs)} EU sources — does auditing pay?')
    ax.legend(frameon=False, fontsize=10); ax.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs('figures/scale', exist_ok=True)
    out = f'figures/scale/far_targets_{REGION}.pdf'
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.pdf', '.png'), dpi=140, bbox_inches='tight')
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
