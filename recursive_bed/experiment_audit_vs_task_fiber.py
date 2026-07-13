"""
Audit vs. task on REAL data, auditing the RIGHT model component.

The geodesic experiment (experiment_audit_vs_task_real.py) showed that auditing
the additive(d/100) model does NOT pay on real pings: its dominant error is
distance-dependent routing overhead (structural), which per-node-offset audits
cannot fix, so the optimal audit share is 0%.

Here we swap the base RTT term d/100 for the fiber-floor atlas (internet_gmaps
FloorEstimator via FiberFloorRtt). The fiber floor captures the distance-
dependent structure, so the RESIDUAL over the floor is much closer to a
per-node constant slack — which IS what audits pin. FINDING (2026-07-10): the fiber base breaks the plateau and beats NN at high
budget (task-only fiber 629 vs geodesic 926 vs NN 726 km at b=175) — fixing
the base model is the real lever. But dedicated VP-VP AUDIT pings STILL do not
pay: audit+task(fiber) is worse than task-only(fiber), and the optimal audit
share stays 0% under BOTH bases and across target counts (tested down to 6).
Why: VP-VP audits pin only the per-node SOURCE offsets, but (a) task pings
already pool enough to identify those, and (b) the dominant reducible error is
either base-model STRUCTURE (fixed by external fiber data, not by spending
pings) or non-additive / target-side residual that VP-VP audits cannot reach.
The audit framework is vindicated by correctly allocating ~0% here: expected
audit value is genuinely low on this data. Audits pay in the synthetic world
because the auditable component was made dominant BY CONSTRUCTION.

Same submesh / ordering / estimator as the geodesic experiment; the only change
is rtt_model. Output: figures/audit_vs_task_fiber.pdf (+ .png), three panels:
  A  error vs budget: fiber breaks the plateau & beats NN; audits add nothing
  B  "confidently wrong" under fiber (overconfidence persists)
  C  allocation sweep: optimum stays 0% under both bases
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
from glob import glob

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'internet_gmaps'))

from experiment_audit_vs_task_real import (load_submesh, budget_sweep,
                                           allocation_sweep, COVERAGE_CAP,
                                           N_TARGETS)
from probabilistic_helpers import FiberFloorRtt


def make_fiber(mesh, slope=1.3):
    """FiberFloorRtt over the plain fiber floor (FloorEstimator — no transit
    policy, so no reverse_geocoder dependency), VP rows = the submesh sources."""
    from fiber_graph import FiberGraph
    from floor_query import FloorEstimator
    npz = np.load(sorted(glob('internet_gmaps/data/graph_*.npz'))[-1])
    graph = FiberGraph(
        npz['node_lat'], npz['node_lon'], npz['edge_src'], npz['edge_dst'],
        npz['edge_rtt_ms'],
        edge_feature=npz['edge_feature'] if 'edge_feature' in npz else None,
        feature_names=tuple(npz['feature_names']) if 'feature_names' in npz else (),
    )
    srcs = mesh['sources']
    vlat = np.array([mesh['src_loc'][s][0] for s in srcs])
    vlon = np.array([mesh['src_loc'][s][1] for s in srcs])
    est = FloorEstimator(graph, vlat, vlon)
    return FiberFloorRtt(estimator=est,
                         vp_locs=[mesh['src_loc'][s] for s in srcs], slope=slope)


def main():
    mesh = load_submesh()
    fiber = make_fiber(mesh, slope=1.3)
    seeds = list(range(10))
    budgets = [20, 30, 40, 50, 60, 75, 90, 110, 130, 150, 175]

    print("geodesic budget sweep...")
    geo = budget_sweep(mesh, seeds, budgets, audit_fraction=0.3, rtt_model=None)
    print("fiber budget sweep...")
    fib = budget_sweep(mesh, seeds, budgets, audit_fraction=0.3, rtt_model=fiber)

    alloc_budget = 120
    fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print("geodesic allocation sweep...")
    alloc_geo = allocation_sweep(mesh, seeds, alloc_budget, fractions, rtt_model=None)
    print("fiber allocation sweep...")
    alloc_fib = allocation_sweep(mesh, seeds, alloc_budget, fractions, rtt_model=fiber)

    print("\nbudget | nn   | task_geo | task_fib | audit_fib")
    for i, B in enumerate(budgets):
        print(f"{B:6d} | {geo['nn']['err'][i]:4.0f} | {geo['task']['err'][i]:8.0f} | "
              f"{fib['task']['err'][i]:8.0f} | {fib['audit']['err'][i]:8.0f}")
    print("\ngeodesic alloc (audit%->km):",
          {int(f*100): round(float(e)) for f, e in zip(fractions, alloc_geo)})
    print("fiber    alloc (audit%->km):",
          {int(f*100): round(float(e)) for f, e in zip(fractions, alloc_fib)})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    C_NN, C_GEO, C_FIB, C_AUD = '#6a994e', '#c9a227', '#d1495b', '#2e86ab'
    b = np.array(budgets)
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # --- A: error vs budget ---
    axA.plot(b, geo['nn']['err'], '--^', color=C_NN, lw=2,
             label='random + nearest-neighbor')
    axA.plot(b, geo['task']['err'], '-o', color=C_GEO, lw=2,
             label='task-only, geodesic base')
    axA.plot(b, fib['task']['err'], '-o', color=C_FIB, lw=2,
             label='task-only, FIBER base')
    axA.plot(b, fib['audit']['err'], '-s', color=C_AUD, lw=2,
             label='audit + task, FIBER base')
    axA.set_xlabel('total budget (pings)')
    axA.set_ylabel('mean localization error (km)')
    axA.set_title('A  Fiber base breaks the plateau & beats NN; audits add nothing')
    axA.legend(frameon=False, fontsize=8.5)
    axA.grid(alpha=0.25)

    # --- B: confidently wrong, under fiber ---
    axB.fill_between(b, fib['task']['rep'], fib['task']['err'],
                     color=C_FIB, alpha=0.12, label='task-only gap (fiber)')
    axB.plot(b, fib['task']['err'], '-o', color=C_FIB, lw=2,
             label='task-only fiber: TRUE')
    axB.plot(b, fib['task']['rep'], '--o', color=C_FIB, lw=1.6, mfc='white',
             label='task-only fiber: REPORTED')
    axB.plot(b, fib['audit']['err'], '-s', color=C_AUD, lw=2,
             label='audit+task fiber: TRUE')
    axB.plot(b, fib['audit']['rep'], '--s', color=C_AUD, lw=1.6, mfc='white',
             label='audit+task fiber: REPORTED')
    axB.set_xlabel('total budget (pings)')
    axB.set_ylabel('km')
    axB.set_title('B  Overconfidence gap under the fiber model')
    axB.legend(frameon=False, fontsize=8)
    axB.grid(alpha=0.25)

    # --- C: allocation, geodesic vs fiber (the flip) ---
    fp = np.array(fractions) * 100
    axC.plot(fp, alloc_geo, '-o', color=C_GEO, lw=2, label='geodesic base')
    axC.plot(fp, alloc_fib, '-o', color=C_AUD, lw=2, label='fiber base')
    for arr, col in ((alloc_geo, C_GEO), (alloc_fib, C_AUD)):
        fbest = fractions[int(np.nanargmin(arr))]
        axC.axvline(fbest * 100, color=col, ls=':', alpha=0.7)
    axC.set_xlabel('share of budget spent on AUDIT (%)')
    axC.set_ylabel('mean localization error (km)')
    axC.set_title('C  Audits never pay here: optimum stays 0% (both bases)')
    axC.legend(frameon=False, fontsize=9)
    axC.grid(alpha=0.25)

    fig.suptitle('Audit vs. task on REAL data — the lever is the base MODEL '
                 '(fiber floor), not runtime VP–VP audit pings', fontsize=12.5, y=1.02)
    fig.tight_layout()
    os.makedirs('figures/real_audit', exist_ok=True)
    out = 'figures/real_audit/audit_vs_task_fiber.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=140, bbox_inches='tight')
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
