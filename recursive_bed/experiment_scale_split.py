"""
Scale run across ALL anchors: split the dense RIPE anchor mesh into a SOURCE
set and a (disjoint) TARGET set, then compare all strategies at scale.

Uses the full dense anchor mesh (global, no region/min-distance restriction):
350 sources x 500 targets, all-pairs. Random selection (the greedy per-candidate
refit doesn't scale to 500 targets); estimators = additive on geodesic vs
(gridded) fiber base, task-only vs audit+task, plus the NN baseline.

Panels:
  A  error vs budget — NN, task/audit x geodesic/fiber (does auditing pay at scale?)
  B  allocation sweep — optimal audit share, geodesic vs fiber

Output: figures/scale_split.pdf (+ .png)
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
from gridded_fiber import make_gridded_fiber

# Split src/dst from the anchor mesh (global, all-pairs). Counts overridable
# via argv: <n_src> <n_dst> [min_tgt_cov].  A low min_tgt_cov pulls in the
# sparse edge-probe tail as targets (e.g. 1000 dst from 50 sources).
import sys as _sys
E.REGION = (-90.0, 90.0, -180.0, 180.0)
E.MIN_SRC_DIST_KM = 0.0
E.N_SOURCES = int(_sys.argv[1]) if len(_sys.argv) > 1 else 350
E.N_TARGETS = int(_sys.argv[2]) if len(_sys.argv) > 2 else 500
E.MIN_TGT_COV = int(_sys.argv[3]) if len(_sys.argv) > 3 else 20
E.POOL = E.N_SOURCES + E.N_TARGETS + 400
E.COVERAGE_CAP = 10000


def main():
    mesh = E.load_submesh()
    fiber = make_gridded_fiber(mesh, res_deg=0.5, slope=1.3)   # global grid
    fiber_seeds = list(range(2))
    seeds = list(range(3))
    nt = len(mesh['targets'])
    budgets = [p * nt for p in (5, 10, 15, 20, 30)]      # pings-per-target grid
    fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    alloc_budget = 15 * nt

    print(f"submesh: {len(mesh['sources'])} sources x {nt} targets, "
          f"{len(mesh['task_edges'])} pairs; budgets {budgets}")
    print("geodesic budget sweep...")
    geo = E.budget_sweep(mesh, seeds, budgets, audit_fraction=0.3, rtt_model=None)
    print("fiber budget sweep...")
    fib = E.budget_sweep(mesh, fiber_seeds, budgets, audit_fraction=0.3, rtt_model=fiber)
    print("geodesic allocation sweep...")
    ag = E.allocation_sweep(mesh, seeds, alloc_budget, fractions, rtt_model=None)
    print("fiber allocation sweep...")
    af = E.allocation_sweep(mesh, fiber_seeds, alloc_budget, fractions, rtt_model=fiber)

    print("\nbudget |  NN  | task_geo audit_geo | task_fib audit_fib")
    for i, B in enumerate(budgets):
        print(f"{B:6d} | {geo['nn']['err'][i]:4.0f} | {geo['task']['err'][i]:8.0f} "
              f"{geo['audit']['err'][i]:9.0f} | {fib['task']['err'][i]:8.0f} "
              f"{fib['audit']['err'][i]:9.0f}")
    print("geodesic alloc:", {int(f*100): round(float(e)) for f, e in zip(fractions, ag)})
    print("fiber    alloc:", {int(f*100): round(float(e)) for f, e in zip(fractions, af)})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    C_NN, C_TG, C_AG, C_TF, C_AF = '#6a994e', '#c9a227', '#e9c46a', '#d1495b', '#2e86ab'
    b = np.array(budgets)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.2))

    axA.plot(b, geo['nn']['err'], '--^', color=C_NN, lw=2, label='random + NN')
    axA.plot(b, geo['task']['err'], '-o', color=C_TG, lw=2, label='task-only · geodesic')
    axA.plot(b, geo['audit']['err'], '-s', color=C_AG, lw=1.8, label='audit+task · geodesic')
    axA.plot(b, fib['task']['err'], '-o', color=C_TF, lw=2, label='task-only · FIBER')
    axA.plot(b, fib['audit']['err'], '-s', color=C_AF, lw=1.8, label='audit+task · FIBER')
    axA.set_xlabel('total budget (pings)'); axA.set_ylabel('mean localization error (km)')
    axA.set_title(f"A  {len(mesh['sources'])} src x {len(mesh['targets'])} dst — error vs budget")
    axA.legend(frameon=False, fontsize=8.5); axA.grid(alpha=0.25)

    fp = np.array(fractions) * 100
    axB.plot(fp, ag, '-o', color=C_TG, lw=2, label='geodesic base')
    axB.plot(fp, af, '-o', color=C_AF, lw=2, label='fiber base')
    for arr, col in ((ag, C_TG), (af, C_AF)):
        axB.axvline(fractions[int(np.nanargmin(arr))] * 100, color=col, ls=':', alpha=0.7)
    axB.set_xlabel('share of budget spent on AUDIT (%)')
    axB.set_ylabel('mean localization error (km)')
    axB.set_title(f'B  Optimal audit share (budget={alloc_budget})')
    axB.legend(frameon=False, fontsize=9); axB.grid(alpha=0.25)

    fig.suptitle('Full anchor mesh split src/dst — how strategies perform at scale',
                 fontsize=13, y=1.0)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = f"figures/scale_split_{len(mesh['sources'])}x{nt}.pdf"
    fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.pdf', '.png'), dpi=140, bbox_inches='tight')
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
