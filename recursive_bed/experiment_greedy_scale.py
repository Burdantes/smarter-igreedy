"""
Robust greedy vs random selection, using the project's real
Iterative_Greedy_Geolocator (batch, region-based — no fragile incremental
estimator). Answers the feedback-loop question at scale, trustworthily:
does model-guided GREEDY selection beat RANDOM selection, and does the fiber
base change the picture?

Anchor mesh split into sources (VPs) x targets (dst). Strategies:
  random + NN            baseline (lowest-RTT source's location)
  random + additive      random selection, additive estimator (geodesic base)
  greedy  (geodesic)     Iterative_Greedy_Geolocator, ADDITIVE, phased
  greedy  (fiber)        same, with the fiber-floor base (exact FiberFloorRtt)

Output: figures/greedy_scale.pdf (+ .png).  Must be run as a script (the greedy
uses multiprocessing).
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
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'internet_gmaps'))

import experiment_audit_vs_task_real as E
from experiment_audit_vs_task_fiber import make_fiber
from iterative_greedy_geolocator import Iterative_Greedy_Geolocator
from feasible_region_maintainer import ADDITIVE
from utils import get_distance

# src/dst split from the anchor mesh. Counts overridable via argv: <n_src> <n_dst>.
# Global, no min-distance exclusion => use the full anchor mesh (all-pairs).
E.REGION = (-90.0, 90.0, -180.0, 180.0)
E.MIN_SRC_DIST_KM = 0.0
E.N_SOURCES = int(sys.argv[1]) if len(sys.argv) > 1 else 60
E.N_TARGETS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
E.POOL = E.N_SOURCES + E.N_TARGETS + 300
E.MIN_TGT_COV = 15
E.COVERAGE_CAP = 10000


def build_data(mesh):
    a2l = {**{s: mesh['src_loc'][s] for s in mesh['sources']},
           **{t: mesh['tgt_loc'][t] for t in mesh['targets']}}
    llm = {}
    for s, t in mesh['task_edges']:
        llm.setdefault(s, {})[t] = [mesh['rtt'](s, t)]
    return {'address_to_loc': a2l, 'loc_loc_meas': llm}


def score(mesh, est):
    e = [get_distance(est[t], mesh['tgt_loc'][t]) for t in mesh['targets'] if t in est]
    return (float(np.mean(e)), float(np.median(e))) if e else (np.nan, np.nan)


def greedy_errors(mesh, data, budgets, rtt_model):
    """Fresh greedy per budget (clean); returns {budget: (mean, median)}."""
    out = {}
    for B in budgets:
        g = Iterative_Greedy_Geolocator(region_mode=ADDITIVE, selection='phased',
                                        rtt_model=rtt_model, max_workers=4,
                                        name='greedy')
        g.set_data(data)
        g.solve()
        g.measurements(B)
        out[B] = score(mesh, g.get_current_estimates())
        print(f"    greedy B={B}: mean {out[B][0]:.0f} median {out[B][1]:.0f}")
    return out


def main():
    mesh = E.load_submesh()
    data = build_data(mesh)
    seeds = [0, 1, 2]
    nt = len(mesh['targets'])
    budgets = [p * nt for p in (8, 16, 25, 40)]   # pings-per-target grid
    print(f"{len(mesh['sources'])} sources x {nt} targets; budgets {budgets}")

    # random baselines (averaged over seeds) via the batch estimator
    def rand_sweep(rtt_model):
        sw = E.budget_sweep(mesh, seeds, budgets, audit_fraction=0.0, rtt_model=rtt_model)
        return sw
    print("random + additive (geodesic)...")
    rgeo = rand_sweep(None)
    print("random + NN is inside the sweep (nn arm)")

    print("greedy (geodesic)...")
    g_geo = greedy_errors(mesh, data, budgets, None)
    print("greedy (fiber)...")
    fiber = make_fiber(mesh)
    g_fib = greedy_errors(mesh, data, budgets, fiber)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    b = np.array(budgets)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(b, rgeo['nn']['err'], '--^', color='#6a994e', lw=2, label='random + NN')
    ax.plot(b, rgeo['task']['err'], '-o', color='#c9a227', lw=2, label='random + additive (geodesic)')
    ax.plot(b, [g_geo[B][0] for B in budgets], '-s', color='#d1495b', lw=2, label='GREEDY (geodesic)')
    ax.plot(b, [g_fib[B][0] for B in budgets], '-s', color='#2e86ab', lw=2, label='GREEDY (fiber)')
    ax.set_xlabel('total budget (pings)'); ax.set_ylabel('mean localization error (km)')
    ax.set_title(f"Real greedy geolocator vs random — {len(mesh['sources'])} src x "
                 f"{len(mesh['targets'])} dst")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = f"figures/greedy_scale_{len(mesh['sources'])}x{nt}.pdf"
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.pdf', '.png'), dpi=140, bbox_inches='tight')
    print(f"wrote {out}")
    print("\nbudget | nn   | rand_add | greedy_geo | greedy_fib")
    for i, B in enumerate(budgets):
        print(f"{B:6d} | {rgeo['nn']['err'][i]:4.0f} | {rgeo['task']['err'][i]:8.0f} | "
              f"{g_geo[B][0]:10.0f} | {g_fib[B][0]:10.0f}")


if __name__ == '__main__':
    main()
